# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Constraint-queue reconciliation for graph-mode quantization annotation.

The atomic unit is a :class:`NodeSlot` — a specific input or output slot on
an fx node. Every fx edge has two slots (producer's OUTPUT + consumer's
INPUT), and each slot carries a :class:`ProvisionalQSpec` holding sparse
per-field values with per-field priorities.

Reconciliation is driven by a queue of :class:`Constraint`s. Each
Constraint relates a set of slots and, when applied, either broadcasts a
per-field decision across the slots (:class:`ShareFields`) or additionally
merges the slots' :class:`ProvisionalQSpec` objects into one shared
instance (:class:`ShareObserverInstance`). Constraint generation is
pattern-driven per op family, and re-fires whenever a touched slot's
state changes.

Convergence: a Constraint that finds the state already-satisfying is a
no-op. When it does mutate state, it only ever *raises* per-field
priority (i.e. the priority-number of a field can only decrease). Because
priorities are bounded below by zero and the state space is finite, the
drain terminates in O(slots × fields × max_priority) iterations.

Pipeline (invoked by :meth:`_AnnotationHandler.annotate`):

    1. Pattern matching (unchanged; performed upstream).
    2. Build the initial per-slot :class:`ProvisionalQSpec` map from
       winning configs plus op-intrinsic overrides (sigmoid/tanh forcing
       ``qscheme=per_tensor_affine``, etc.).
    3. Seed the constraint queue by running
       :func:`_generate_constraints_for_node` over every node.
    4. Drain the queue. Each applied constraint may mutate the state and
       return slots whose state changed; re-enqueue constraints for the
       nodes touching those slots.
    5. Serialize the final state onto per-node
       :class:`QuantizationAnnotation` fields — slots sharing a
       :class:`ProvisionalQSpec` object become one anchor's concrete spec
       plus :class:`SharedQuantizationSpec` back-edges.
"""

from __future__ import annotations

import enum
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.fx as fx
from torch.fx.passes.utils.matcher_utils import InternalMatch
from torch.fx.passes.utils.source_matcher_utils import SourcePartition
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec as TorchAOQuantizationSpec,
    SharedQuantizationSpec as _SharedQuantizationSpec,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

from coreai_opt._utils.config_utils import ALL_TENSORS as _ALL_TENSORS
from coreai_opt._utils.fx_utils import (
    get_local_state_name as _get_local_state_name,
    is_coreai_compressed_state_node as _is_state_node,
)
from coreai_opt.quantization.config import OpQuantizerConfig

from ._annotation_config import AnnotationConfig
from ._annotation_utils import (
    _always_affine_ops,
    _fixed_q_params_ops,
    _get_call_function_node_from_partition,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Atomic units.
# ---------------------------------------------------------------------------


class SlotKind(enum.Enum):
    """Which side of a node a slot lives on."""

    INPUT = enum.auto()
    OUTPUT = enum.auto()


@dataclass(frozen=True)
class NodeSlot:
    """A single quantization decision point on a node.

    Each fx node contributes one ``OUTPUT`` slot (``arg_index=0``) and one
    ``INPUT`` slot per unique input in ``node.all_input_nodes``
    (``arg_index`` tracks position). Slots are the reconciliation atom.
    """

    node: fx.Node
    kind: SlotKind
    arg_index: int


class FieldName(enum.Enum):
    """Fields on a :class:`ProvisionalQSpec`.

    Each field has its own reconciliation policy (see :func:`_reconcile_field`).
    """

    DTYPE = enum.auto()
    QSCHEME = enum.auto()  # combined symmetry × granularity — one lattice-joined field
    CH_AXIS = enum.auto()
    QUANT_MIN = enum.auto()
    QUANT_MAX = enum.auto()
    IS_DYNAMIC = enum.auto()
    OBSERVER_CLASS = enum.auto()


_ALL_FIELDS: frozenset[FieldName] = frozenset(FieldName)


@dataclass(frozen=True)
class FieldValue:
    """A value proposed for one field on one slot, plus its priority.

    Priority follows the sorted-list convention: **lower priority number =
    higher priority** (top of the sort). Reconciliation only ever raises
    priority-ness (lowers the number).
    """

    value: Any
    priority: int


@dataclass
class ProvisionalQSpec:
    """Mutable per-observer state.

    Multiple slots may reference one instance — that's how
    :class:`ShareObserverInstance` expresses sharing: two slots point at
    the same Python object, so any field mutation is seen by both. Object
    identity IS the sharing relation; there's no separate side-list.
    """

    fields: dict[FieldName, FieldValue] = field(default_factory=dict)


State = dict[NodeSlot, ProvisionalQSpec]


# ---------------------------------------------------------------------------
# Constraints.
# ---------------------------------------------------------------------------


class Constraint(ABC):
    """A relation over one or more slots.

    :meth:`apply` reads the current state, mutates it to enforce the
    constraint, and returns the set of slots whose state changed. Returning
    an empty set means the constraint was already satisfied (drives
    convergence).
    """

    @property
    @abstractmethod
    def slots(self) -> frozenset[NodeSlot]: ...

    @abstractmethod
    def apply(self, state: State) -> set[NodeSlot]: ...


@dataclass(frozen=True)
class ShareFields(Constraint):
    """Every slot in ``slots`` must agree on each of ``fields``.

    Reconciliation is per-field (see :func:`_reconcile_field`). The winning
    value is broadcast to every slot in the group; slots that had no
    proposal for a field pick up the winner. Priority of the reconciled
    value inherits the highest priority (lowest priority-number) among the
    contributing proposals.
    """

    _slots: frozenset[NodeSlot]
    fields: frozenset[FieldName]

    @property
    def slots(self) -> frozenset[NodeSlot]:
        return self._slots

    def apply(self, state: State) -> set[NodeSlot]:
        changed: set[NodeSlot] = set()
        for f in self.fields:
            reconciled = _reconcile_field(f, self._slots, state)
            if reconciled is None:
                continue
            for slot in self._slots:
                spec = _get_or_create(state, slot)
                current = spec.fields.get(f)
                if _field_value_stronger(reconciled, current):
                    spec.fields[f] = reconciled
                    changed.add(slot)
        return changed


@dataclass(frozen=True)
class ShareObserverInstance(Constraint):
    """Every slot in ``slots`` must reference the same
    :class:`ProvisionalQSpec` object (same observer instance at runtime).

    Applying:

        1. Find every slot transitively sharing a ProvisionalQSpec with any
           slot in ``self.slots`` (i.e. widen the group to include
           previously-shared siblings).
        2. Reconcile every field across the widened group.
        3. Replace every widened slot's state entry with one shared
           :class:`ProvisionalQSpec` holding the reconciled fields.

    Because sharing is expressed by object identity, subsequent field
    mutations on any member automatically propagate to every sharer.
    """

    _slots: frozenset[NodeSlot]

    @property
    def slots(self) -> frozenset[NodeSlot]:
        return self._slots

    def apply(self, state: State) -> set[NodeSlot]:
        # Widen: pull in every slot already sharing with any input slot.
        target_ids = {id(state[s]) for s in self._slots if s in state}
        widened: set[NodeSlot] = set(self._slots)
        for slot, spec in state.items():
            if id(spec) in target_ids:
                widened.add(slot)

        # Reconcile fields across the widened group.
        reconciled_fields: dict[FieldName, FieldValue] = {}
        for f in _ALL_FIELDS:
            rv = _reconcile_field(f, frozenset(widened), state)
            if rv is not None:
                reconciled_fields[f] = rv

        # Decide whether anything actually changed: any slot's spec differs
        # from the reconciled fields, or two slots don't share identity yet.
        changed: set[NodeSlot] = set()
        first_ref: ProvisionalQSpec | None = None
        for slot in widened:
            existing = state.get(slot)
            already_reconciled = existing is not None and existing.fields == reconciled_fields
            if first_ref is None:
                if already_reconciled:
                    first_ref = existing
                else:
                    first_ref = ProvisionalQSpec(fields=dict(reconciled_fields))
                    state[slot] = first_ref
                    changed.add(slot)
            else:
                if state.get(slot) is not first_ref:
                    state[slot] = first_ref
                    changed.add(slot)
                if not already_reconciled:
                    # first_ref's fields need updating too — do it once when
                    # we discover the discrepancy.
                    if first_ref.fields != reconciled_fields:
                        first_ref.fields = dict(reconciled_fields)
                        # Every widened slot now sees the new fields; mark
                        # all as changed so downstream generators re-run.
                        changed.update(widened)
        # Corner case: single-slot ShareObserverInstance whose state was
        # already correct.
        if first_ref is not None and first_ref.fields != reconciled_fields:
            first_ref.fields = dict(reconciled_fields)
            changed.update(widened)
        return changed


# ---------------------------------------------------------------------------
# Per-field reconciliation policies.
# ---------------------------------------------------------------------------


class ReconciliationError(RuntimeError):
    """Raised when a constraint can't be satisfied. Fatal cases:

    * ``IS_DYNAMIC`` disagreement across slots.
    * Incompatible ``CH_AXIS`` values across per_channel slots.
    """


def _reconcile_field(
    fname: FieldName, slots: frozenset[NodeSlot], state: State
) -> FieldValue | None:
    """Compute the reconciled ``FieldValue`` for one field across ``slots``.

    Returns ``None`` if no slot has a proposal for the field (nothing to
    do). Raises :class:`ReconciliationError` on infeasible fields.
    """
    proposals = [state[s].fields[fname] for s in slots if s in state and fname in state[s].fields]
    if not proposals:
        return None

    policy = _FIELD_POLICY[fname]
    return policy(proposals)


def _priority_min(proposals: Sequence[FieldValue]) -> int:
    """Highest priority (lowest priority-number) among the proposals."""
    return min(p.priority for p in proposals)


def _policy_priority_wins(proposals: Sequence[FieldValue]) -> FieldValue:
    """Value from the highest-priority proposal wins. Ties: first encountered."""
    winner = min(proposals, key=lambda p: p.priority)
    return FieldValue(value=winner.value, priority=_priority_min(proposals))


def _policy_must_agree(proposals: Sequence[FieldValue]) -> FieldValue:
    """All values must be equal, else :class:`ReconciliationError`."""
    values = {p.value for p in proposals}
    if len(values) > 1:
        raise ReconciliationError(
            f"Incompatible values across slots: {sorted(str(v) for v in values)}. "
            f"Proposals: {[(p.value, p.priority) for p in proposals]}"
        )
    return FieldValue(value=next(iter(values)), priority=_priority_min(proposals))


def _policy_qscheme_lattice(proposals: Sequence[FieldValue]) -> FieldValue:
    """Lattice join: symmetric ⊂ affine; per_tensor ⊂ per_channel. Looser wins."""
    qschemes = [p.value for p in proposals if p.value is not None]
    if not qschemes:
        return FieldValue(value=None, priority=_priority_min(proposals))
    joined = _join_qscheme(set(qschemes))
    return FieldValue(value=joined, priority=_priority_min(proposals))


def _policy_range_min(proposals: Sequence[FieldValue]) -> FieldValue:
    """Union — take the smallest quant_min across proposals."""
    values = [p.value for p in proposals if p.value is not None]
    if not values:
        return FieldValue(value=None, priority=_priority_min(proposals))
    return FieldValue(value=min(values), priority=_priority_min(proposals))


def _policy_range_max(proposals: Sequence[FieldValue]) -> FieldValue:
    """Union — take the largest quant_max across proposals."""
    values = [p.value for p in proposals if p.value is not None]
    if not values:
        return FieldValue(value=None, priority=_priority_min(proposals))
    return FieldValue(value=max(values), priority=_priority_min(proposals))


_FIELD_POLICY: dict[FieldName, Any] = {
    FieldName.DTYPE: _policy_priority_wins,
    FieldName.QSCHEME: _policy_qscheme_lattice,
    FieldName.CH_AXIS: _policy_must_agree,
    FieldName.QUANT_MIN: _policy_range_min,
    FieldName.QUANT_MAX: _policy_range_max,
    FieldName.IS_DYNAMIC: _policy_must_agree,
    FieldName.OBSERVER_CLASS: _policy_priority_wins,
}


def _field_value_stronger(new: FieldValue, current: FieldValue | None) -> bool:
    """Return True iff writing ``new`` would advance state.

    "Advance" = value differs, or value equals but priority is stronger
    (lower priority-number). Enables no-op detection for convergence.
    """
    if current is None:
        return True
    if new.value != current.value:
        return True
    return new.priority < current.priority


_QSCHEME_IS_SYMMETRIC: dict[Any, bool] = {
    None: True,
    torch.per_tensor_symmetric: True,
    torch.per_tensor_affine: False,
    torch.per_channel_symmetric: True,
    torch.per_channel_affine: False,
}
_QSCHEME_IS_PER_CHANNEL: dict[Any, bool] = {
    None: False,
    torch.per_tensor_symmetric: False,
    torch.per_tensor_affine: False,
    torch.per_channel_symmetric: True,
    torch.per_channel_affine: True,
}
_PER_CHANNEL_SCHEMES = (torch.per_channel_symmetric, torch.per_channel_affine)


def _join_qscheme(qschemes: set[Any]) -> Any:
    is_sym = all(_QSCHEME_IS_SYMMETRIC.get(q, False) for q in qschemes if q is not None)
    is_per_channel = any(_QSCHEME_IS_PER_CHANNEL.get(q, False) for q in qschemes)
    if is_per_channel:
        return torch.per_channel_symmetric if is_sym else torch.per_channel_affine
    return torch.per_tensor_symmetric if is_sym else torch.per_tensor_affine


# ---------------------------------------------------------------------------
# Slot helpers.
# ---------------------------------------------------------------------------


def _output_slot(node: fx.Node) -> NodeSlot:
    return NodeSlot(node=node, kind=SlotKind.OUTPUT, arg_index=0)


def _input_slot(consumer: fx.Node, arg_index: int) -> NodeSlot:
    return NodeSlot(node=consumer, kind=SlotKind.INPUT, arg_index=arg_index)


def _input_slot_for_producer(consumer: fx.Node, producer: fx.Node) -> NodeSlot:
    for i, p in enumerate(consumer.all_input_nodes):
        if p is producer:
            return _input_slot(consumer, i)
    raise ValueError(f"{producer.name!r} is not an input of {consumer.name!r}")


def _enumerate_slots(model: fx.GraphModule) -> Iterator[NodeSlot]:
    for node in model.graph.nodes:
        yield _output_slot(node)
        for i in range(len(node.all_input_nodes)):
            yield _input_slot(node, i)


def _get_or_create(state: State, slot: NodeSlot) -> ProvisionalQSpec:
    if slot not in state:
        state[slot] = ProvisionalQSpec()
    return state[slot]


# ---------------------------------------------------------------------------
# Context bundle (Phase-1/2 outputs).
# ---------------------------------------------------------------------------


@dataclass
class _AnnotationContext:
    """Inputs to the reconciliation pipeline.

    Attributes:
        winning_configs (dict[fx.Node, OpQuantizerConfig]): Highest-priority
            config for each covered node.
        node_priorities (dict[fx.Node, int]): Position of each covered
            node in the sorted-list traversal. Lower = higher priority.
        pattern_groups (dict[fx.Node, frozenset[fx.Node]]): For each covered
            node, the set of fx nodes belonging to the same winning
            pattern match. Used to skip *internal* slots (both endpoints
            in the same pattern don't get their own annotations).
        shared_observer_nodes (set[fx.Node]): Cached set of shared-observer
            pattern nodes — used by constraint generators.
    """

    winning_configs: dict[fx.Node, OpQuantizerConfig]
    node_priorities: dict[fx.Node, int]
    pattern_groups: dict[fx.Node, frozenset[fx.Node]]
    shared_observer_nodes: set[fx.Node]


# ---------------------------------------------------------------------------
# Initial state construction.
# ---------------------------------------------------------------------------


def build_initial_state(model: fx.GraphModule, ctx: _AnnotationContext) -> State:
    """Build the per-slot :class:`ProvisionalQSpec` map from configs + intrinsics.

    For each covered node, emit field values on:

        * Each INPUT slot whose producer is *outside* the node's pattern
          group (internal edges skip both sides). Values come from
          ``op_input_spec[arg_index]`` (or ``op_state_spec[state_name]``
          if the producer is a state node).
        * The OUTPUT slot, unless every consumer is inside the pattern
          group. Values come from ``op_output_spec[0]``. Then, if the
          node's target triggers an op-intrinsic override
          (sigmoid/tanh/hardsigmoid → per_tensor_affine; relu/relu6 →
          per_tensor_affine), overwrite ``QSCHEME`` at the same priority.

    Every field value's priority is the node's ``ctx.node_priorities`` entry.
    """
    state: State = {}
    node_to_annotation_config: dict[fx.Node, AnnotationConfig] = {
        node: AnnotationConfig.from_quantizer_config(cfg)
        for node, cfg in ctx.winning_configs.items()
    }

    for node, cfg in node_to_annotation_config.items():
        priority = ctx.node_priorities[node]
        pattern = ctx.pattern_groups.get(node, frozenset({node}))

        # INPUT slots.
        for arg_index, producer in enumerate(node.all_input_nodes):
            if producer in pattern:
                continue  # internal edge — skip
            slot = _input_slot(node, arg_index)
            if _is_state_node(producer):
                # Reject configs that target a state tensor via op_input_spec:
                # weights and biases must be configured via op_state_spec.
                if arg_index in cfg.op_input_spec:
                    raise RuntimeError(
                        f"Config is attempting to set op_input_spec idx {arg_index}, "
                        f"but the input is a state tensor (node: {producer.name}). "
                        f"Use op_state_spec to configure state inputs instead.\n"
                        f"op_input_spec: {cfg.op_input_spec}"
                    )
                spec = _lookup_state_spec(cfg, producer)
            else:
                spec = _lookup_by_key(cfg.op_input_spec, arg_index)
            if spec is None:
                continue
            _populate_fields_from_spec(state, slot, spec, priority)

        # OUTPUT slot — skipped when the pattern absorbs every consumer.
        if node.users and all(c in pattern for c in node.users):
            continue
        output_slot = _output_slot(node)
        out_spec = _lookup_by_key(cfg.op_output_spec, 0)
        if out_spec is not None:
            _populate_fields_from_spec(state, output_slot, out_spec, priority)

        # Op-intrinsic qscheme overrides. Baked in at construction — they
        # participate in reconciliation as regular field values.
        intrinsic_qscheme = _op_intrinsic_qscheme(node)
        if intrinsic_qscheme is not None and out_spec is not None:
            spec = _get_or_create(state, output_slot)
            spec.fields[FieldName.QSCHEME] = FieldValue(value=intrinsic_qscheme, priority=priority)

    return state


def _populate_fields_from_spec(
    state: State, slot: NodeSlot, spec: TorchAOQuantizationSpec, priority: int
) -> None:
    """Copy fields from a torchao :class:`QuantizationSpec` into the state."""
    ps = _get_or_create(state, slot)
    ps.fields[FieldName.DTYPE] = FieldValue(value=spec.dtype, priority=priority)
    ps.fields[FieldName.OBSERVER_CLASS] = FieldValue(
        value=spec.observer_or_fake_quant_ctr, priority=priority
    )
    ps.fields[FieldName.IS_DYNAMIC] = FieldValue(value=spec.is_dynamic, priority=priority)
    if spec.quant_min is not None:
        ps.fields[FieldName.QUANT_MIN] = FieldValue(value=spec.quant_min, priority=priority)
    if spec.quant_max is not None:
        ps.fields[FieldName.QUANT_MAX] = FieldValue(value=spec.quant_max, priority=priority)
    if spec.qscheme is not None:
        ps.fields[FieldName.QSCHEME] = FieldValue(value=spec.qscheme, priority=priority)
    if spec.ch_axis is not None:
        ps.fields[FieldName.CH_AXIS] = FieldValue(value=spec.ch_axis, priority=priority)


def _op_intrinsic_qscheme(node: fx.Node) -> Any | None:
    """Return the qscheme forced by op semantics, or ``None``.

    Only qscheme is baked in — the current PyTorch/torchao workaround
    ignores fixed quant_min/max on FixedQParamsQuantizationSpec anyway
    (see ``_annotation_utils.py:289-292`` FIXME). Range/dtype come from
    user configs.
    """
    if node.target in _fixed_q_params_ops:
        return _fixed_q_params_ops[node.target].qscheme
    if node.target in _always_affine_ops:
        return torch.per_tensor_affine
    if node.target in (torch.ops.aten.hardtanh.default, torch.ops.aten.hardtanh_.default):
        if len(node.args) >= 3 and node.args[1] == 0 and node.args[2] == 6:
            return torch.per_tensor_affine
    return None


def _lookup_by_key(spec_map: dict[Any, Any], key: Any) -> Any:
    if key in spec_map:
        return spec_map[key]
    if _ALL_TENSORS in spec_map:
        return spec_map[_ALL_TENSORS]
    return None


def _lookup_state_spec(
    consumer_cfg: AnnotationConfig, state_node: fx.Node
) -> TorchAOQuantizationSpec | None:
    state_name = _get_local_state_name(state_node)
    if state_name is None:
        return None
    return _lookup_by_key(consumer_cfg.op_state_spec, state_name)


# ---------------------------------------------------------------------------
# Constraint generators — dispatched per node target.
# ---------------------------------------------------------------------------


def _generate_constraints_for_node(
    node: fx.Node, state: State, ctx: _AnnotationContext
) -> list[Constraint]:
    """Return every constraint whose scope includes ``node``.

    Rules run against the current state (dynamic peek); when a slot on a
    downstream/upstream node has already been decided, rules can inspect
    the decision and emit stronger constraints.
    """
    constraints: list[Constraint] = []

    # Rule 1 — adjacent-edge sharing. For every fx edge (P, C, i) touching
    # ``node`` (on either side), tie P.OUTPUT and C.INPUT[i] into one
    # observer group when both slots want observation.
    constraints.extend(_adjacent_edge_constraints(node, state, ctx))

    # Rule 2 — shared-observer op (cat/maxpool/flatten/avgpool).
    if node in ctx.shared_observer_nodes:
        constraints.extend(_shared_observer_constraints(node, state, ctx))

    # Rule 3 — shared weight tensor: for a state producer feeding ≥ 2
    # covered consumers, tie every consumer's INPUT slot that reads the
    # state tensor into one shared-observer group.
    if _is_state_node(node) and len(node.users) >= 2:
        constraints.extend(_shared_state_constraints(node, state, ctx))

    return constraints


def _shared_state_constraints(
    state_node: fx.Node, state: State, ctx: _AnnotationContext
) -> list[Constraint]:
    """Constraints for a state (weight/parameter) node with multiple consumers.

    Weight tensors are quantized once at prepare time; all consumers see
    the same quantized tensor at runtime. Tie every covered consumer's
    INPUT slot that reads this state tensor into one shared-observer
    group. Uncovered consumers are ignored (they don't participate in
    quantization at all).
    """
    consumer_slots: list[NodeSlot] = []
    for consumer in state_node.users:
        if consumer not in ctx.winning_configs:
            continue
        try:
            consumer_slots.append(_input_slot_for_producer(consumer, state_node))
        except ValueError:
            continue

    if len(consumer_slots) < 2:
        return []
    if not any(s in state for s in consumer_slots):
        # No consumer wants observation on this weight → no sharing.
        return []
    return [ShareObserverInstance(frozenset(consumer_slots))]


def _adjacent_edge_constraints(
    node: fx.Node, state: State, ctx: _AnnotationContext
) -> list[Constraint]:
    """For every edge touching ``node`` whose endpoints are both covered
    by pattern matches and at least one endpoint has fields in state,
    emit a ShareObserverInstance tying the two slots together.

    Requiring "both endpoints covered" prevents dragging unannotated
    external nodes (placeholders, unmatched ops) into observer groups.
    Requiring "at least one has fields" fires the merge even when only
    the producer or only the consumer holds proposals — matching the
    inheritance behavior of the current code's
    :func:`_fill_input_qspec_map_for_input`, which set a consumer's
    input spec from an annotated producer's output_qspec.

    Internal edges (both endpoints in the same pattern group) are
    skipped — those slots aren't annotated in the first place.
    """
    if node not in ctx.winning_configs:
        # Uncovered nodes (placeholders, unmatched ops) never anchor an
        # adjacent-edge constraint on either side; a covered neighbor that
        # cares about this edge will emit it from its own side.
        return []

    constraints: list[Constraint] = []
    pattern = ctx.pattern_groups.get(node, frozenset({node}))

    # Incoming edges: (producer, node, i).
    for i, producer in enumerate(node.all_input_nodes):
        if producer in pattern:
            continue
        if producer not in ctx.winning_configs:
            continue
        p_out = _output_slot(producer)
        c_in = _input_slot(node, i)
        if p_out in state or c_in in state:
            constraints.append(ShareObserverInstance(frozenset({p_out, c_in})))

    # Outgoing edges: (node, consumer, i).
    for consumer in node.users:
        if consumer in pattern:
            continue
        if consumer not in ctx.winning_configs:
            continue
        try:
            c_in = _input_slot_for_producer(consumer, node)
        except ValueError:
            continue
        p_out = _output_slot(node)
        if p_out in state or c_in in state:
            constraints.append(ShareObserverInstance(frozenset({p_out, c_in})))

    return constraints


def _shared_observer_constraints(
    sh: fx.Node, state: State, ctx: _AnnotationContext
) -> list[Constraint]:
    """Constraints for a shared-observer op (cat/maxpool/avgpool/flatten).

    All emit ``ShareFields`` on ``{DTYPE, IS_DYNAMIC}`` across the op's
    INPUT slots and OUTPUT slot. Concat further decides observer-instance
    sharing based on the current OUTPUT-slot qscheme (peek at state):

        * If OUTPUT.qscheme is per_channel and the ``ch_axis`` matches the
          concat dim, per-channel scales are independent per input →
          only ShareFields on DTYPE/IS_DYNAMIC.
        * Otherwise (per_tensor, or per_channel on a different axis, or
          unknown), all inputs+output must share observer instance.

    Non-concat shared-observer ops (maxpool/avgpool/flatten) are simpler:
    always ShareObserverInstance across their input+output slots.

    Note:
        The whole shared-observer group is gated on **at least one INPUT
        slot having fields**. Shared-observer semantics only activate when
        the op's inputs want observation; if all input slots are empty
        (e.g. flatten configured with ``op_input_spec=None``), the output
        stands alone with whatever its own config says — no cross-slot
        merging.
    """
    input_slots: list[NodeSlot] = [_input_slot(sh, i) for i in range(len(sh.all_input_nodes))]

    if not any(s in state for s in input_slots):
        return []  # no input wants observation → no sharing activates

    constraints: list[Constraint] = []
    output_slot = _output_slot(sh)
    all_slots = frozenset([output_slot] + input_slots)

    # Every shared-observer op ties dtype and is_dynamic.
    constraints.append(
        ShareFields(_slots=all_slots, fields=frozenset({FieldName.DTYPE, FieldName.IS_DYNAMIC}))
    )

    if _is_concat(sh):
        concat_dim = _concat_dim(sh)
        out_qscheme_fv = state.get(output_slot, ProvisionalQSpec()).fields.get(FieldName.QSCHEME)
        out_ch_axis_fv = state.get(output_slot, ProvisionalQSpec()).fields.get(FieldName.CH_AXIS)

        per_channel_along_concat_axis = (
            out_qscheme_fv is not None
            and out_qscheme_fv.value in _PER_CHANNEL_SCHEMES
            and out_ch_axis_fv is not None
            and out_ch_axis_fv.value == concat_dim
        )
        if per_channel_along_concat_axis:
            # Per-channel along concat axis — each input keeps its own scale.
            # No observer-instance sharing among inputs, only dtype (already
            # emitted above).
            pass
        else:
            # Per-tensor, or per-channel on a different axis, or unknown —
            # all inputs+output must share observer instance so scales align.
            constraints.append(ShareObserverInstance(_slots=all_slots))
    else:
        # maxpool / avgpool / flatten — always share observer.
        constraints.append(ShareObserverInstance(_slots=all_slots))

    return constraints


def _is_concat(node: fx.Node) -> bool:
    return node.target in (torch.ops.aten.cat.default, torch.ops.aten.concat.default)


def _concat_dim(node: fx.Node) -> int:
    """Concat dim from ``aten.cat(tensors, dim=?)`` (defaults to 0)."""
    if len(node.args) >= 2:
        return int(node.args[1])
    return int(node.kwargs.get("dim", 0))


# ---------------------------------------------------------------------------
# Drain loop.
# ---------------------------------------------------------------------------


def annotate_via_reconciliation(model: fx.GraphModule, ctx: _AnnotationContext) -> fx.GraphModule:
    """Annotate ``model`` in place using the constraint-queue reconciler."""
    state = build_initial_state(model, ctx)

    queue: deque[Constraint] = deque()
    for node in model.graph.nodes:
        queue.extend(_generate_constraints_for_node(node, state, ctx))

    # Cap iterations as a runaway-loop backstop; convergence is guaranteed
    # by priority monotonicity but a bug in a policy could stall.
    max_iters = 1000 * (1 + sum(1 for _ in _enumerate_slots(model))) * len(_ALL_FIELDS)
    iters = 0
    while queue:
        c = queue.popleft()
        changed = c.apply(state)
        iters += 1
        if iters > max_iters:
            raise RuntimeError(
                f"Constraint drain did not converge after {iters} iterations — "
                f"likely a bug in a reconciliation policy or generator."
            )
        touched_nodes = {slot.node for slot in changed}
        for touched in touched_nodes:
            queue.extend(_generate_constraints_for_node(touched, state, ctx))

    serialize_annotations(model, state, ctx)
    return model


# ---------------------------------------------------------------------------
# Serialization.
# ---------------------------------------------------------------------------


def serialize_annotations(model: fx.GraphModule, state: State, ctx: _AnnotationContext) -> None:
    """Project the final state onto per-node
    :class:`QuantizationAnnotation` fields.

    Slots sharing a :class:`ProvisionalQSpec` object (via
    :class:`ShareObserverInstance`) form one component. Within a component:

        * Anchor = topologically-first slot. Within a single node, INPUT
          slots are preferred over OUTPUT — torchao's
          ``_get_edge_or_node_to_qspec`` iterates ``input_qspec_map``
          entries before ``output_qspec``, so the anchor's observer must
          be registered on the input side to be visible when an output-side
          SharedSpec resolves.
        * Anchor slot → concrete :class:`TorchAOQuantizationSpec`.
        * Other slots → :class:`SharedQuantizationSpec` in Node form for
          OUTPUT anchors, edge form for INPUT anchors.
    """
    # Group slots by ProvisionalQSpec identity.
    groups_by_id: dict[int, list[NodeSlot]] = defaultdict(list)
    ps_by_id: dict[int, ProvisionalQSpec] = {}
    for slot, ps in state.items():
        groups_by_id[id(ps)].append(slot)
        ps_by_id[id(ps)] = ps

    # Compute topo index for anchor selection.
    topo_index: dict[fx.Node, int] = {n: i for i, n in enumerate(model.graph.nodes)}

    def slot_topo_key(slot: NodeSlot) -> tuple[int, int, int]:
        # Within a node, INPUT comes before OUTPUT — torchao's
        # ``_get_edge_or_node_to_qspec`` iterates ``input_qspec_map`` before
        # ``output_qspec``, so the anchor's observer must be registered on
        # the input side to be visible when the output-side SharedSpec is
        # resolved.
        return (
            topo_index.get(slot.node, len(topo_index)),
            0 if slot.kind is SlotKind.INPUT else 1,
            slot.arg_index,
        )

    # Per-node collectors.
    input_maps: dict[fx.Node, dict[fx.Node, Any]] = defaultdict(dict)
    output_specs: dict[fx.Node, Any] = {}

    for gid, slots in groups_by_id.items():
        ps = ps_by_id[gid]
        concrete = _build_concrete_spec(ps)
        if concrete is None:
            continue

        sorted_slots = sorted(slots, key=slot_topo_key)
        anchor = sorted_slots[0]
        shared_ref: Any
        if anchor.kind is SlotKind.OUTPUT:
            shared_ref = _SharedQuantizationSpec(anchor.node)
        else:
            producer = anchor.node.all_input_nodes[anchor.arg_index]
            shared_ref = _SharedQuantizationSpec((producer, anchor.node))

        for slot in sorted_slots:
            value = concrete if slot == anchor else shared_ref
            if slot.kind is SlotKind.OUTPUT:
                output_specs[slot.node] = value
            else:
                producer = slot.node.all_input_nodes[slot.arg_index]
                input_maps[slot.node][producer] = value

    # Write annotations.
    touched: set[fx.Node] = set(input_maps.keys()) | set(output_specs.keys())
    for node in touched:
        annotation = node.meta.get(Q_ANNOTATION_KEY, QuantizationAnnotation())
        input_map = input_maps.get(node)
        if input_map:
            # Backfill None for every positional input that didn't get an
            # explicit spec — torchao's qat_utils positional indexing
            # (``qat_utils.py:547``) IndexErrors if slots are missing.
            full_map: dict[fx.Node, Any] = {}
            for producer in node.all_input_nodes:
                full_map[producer] = input_map.get(producer, None)
            annotation.input_qspec_map = full_map
        if node in output_specs:
            annotation.output_qspec = output_specs[node]
        annotation._annotated = True
        node.meta[Q_ANNOTATION_KEY] = annotation


def _build_concrete_spec(ps: ProvisionalQSpec) -> TorchAOQuantizationSpec | None:
    """Assemble a :class:`TorchAOQuantizationSpec` from a fully-reconciled state.

    Returns ``None`` if required fields (``DTYPE``, ``OBSERVER_CLASS``) are
    missing — that ProvisionalQSpec represents a slot the pipeline chose
    not to observe.
    """
    if FieldName.DTYPE not in ps.fields or FieldName.OBSERVER_CLASS not in ps.fields:
        return None
    return TorchAOQuantizationSpec(
        dtype=ps.fields[FieldName.DTYPE].value,
        observer_or_fake_quant_ctr=ps.fields[FieldName.OBSERVER_CLASS].value,
        quant_min=(
            ps.fields[FieldName.QUANT_MIN].value if FieldName.QUANT_MIN in ps.fields else None
        ),
        quant_max=(
            ps.fields[FieldName.QUANT_MAX].value if FieldName.QUANT_MAX in ps.fields else None
        ),
        qscheme=(ps.fields[FieldName.QSCHEME].value if FieldName.QSCHEME in ps.fields else None),
        ch_axis=(ps.fields[FieldName.CH_AXIS].value if FieldName.CH_AXIS in ps.fields else None),
        is_dynamic=(
            ps.fields[FieldName.IS_DYNAMIC].value if FieldName.IS_DYNAMIC in ps.fields else False
        ),
    )


# ---------------------------------------------------------------------------
# Coverage helper (used by quantizer.py to build the context).
# ---------------------------------------------------------------------------


def _nodes_covered_by(match_info: Any) -> list[fx.Node]:
    """Return every fx node an annotator match annotates."""
    match = match_info.annotator_match

    if isinstance(match, InternalMatch):
        return [node for key, node in match.name_node_map.items() if key in ("mod", "output", "bn")]

    if isinstance(match, tuple) and all(isinstance(p, SourcePartition) for p in match):
        return [_get_call_function_node_from_partition(p) for p in match]

    raise TypeError(
        f"Unknown annotator match type: {type(match).__name__}. Update "
        f"_nodes_covered_by when adding a new pattern family."
    )
