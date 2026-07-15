# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Constraint-queue reconciliation for graph-mode quantization annotation.

Entry point + constraint generators. Types, per-field policies, and
resolution live in sibling modules:

    * ``_qspec_types`` — :class:`NodeSlot`, :class:`FieldName`,
      :class:`ProvisionalQSpec`, :class:`ProvisionalQSpecMap`, etc.
    * ``_qspec_constraints`` — :class:`Constraint` ABC, per-field
      policies, :func:`_reconcile_field`.
    * ``_provisional_qspec_generation`` — :func:`build_initial_state`.
    * ``_qspec_resolution`` — :func:`resolve_qspecs` (per-node
      annotation write).

Pipeline:

    1. Pattern matching (upstream, in ``_AnnotationHandler.annotate``).
    2. :func:`build_initial_state` populates per-slot proposals from
       winning configs + op-intrinsic overrides.
    3. Seed the constraint queue by running
       :func:`_generate_constraints_for_node` over every fx node.
    4. Drain: pop a constraint, apply it, re-enqueue constraints for
       every node whose slots changed. Convergence guaranteed by
       per-field priority monotonicity.
    5. :func:`resolve_qspecs` projects the final map onto per-node
       :class:`QuantizationAnnotation` fields.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch.fx as fx
from torch.fx.passes.utils.matcher_utils import InternalMatch
from torch.fx.passes.utils.source_matcher_utils import SourcePartition

from coreai_opt._utils.fx_utils import is_coreai_compressed_state_node as _is_state_node
from coreai_opt.quantization.config import OpQuantizerConfig

from ._annotation_pattern_registry import (
    SharedObserverModulePattern,
)
from ._annotation_utils import _get_call_function_node_from_partition
from ._provisional_qspec_generation import build_initial_state
from ._qspec_constraints import (
    Constraint,
    ShareObserverInstance,
)
from ._qspec_resolution import resolve_qspecs
from ._qspec_types import (
    NodeSlot,
    ProvisionalQSpecMap,
    SlotKind,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context bundle (Phase-1/2 outputs).
# ---------------------------------------------------------------------------


@dataclass
class _AnnotationContext:
    """Inputs to the reconciliation pipeline.

    Attributes:
        winning_configs (dict[fx.Node, OpQuantizerConfig]): Highest-priority
            config per covered node.
        node_priorities (dict[fx.Node, int]): Position in the sorted-list
            traversal. Lower = higher priority. Baked into every field
            value the pipeline emits.
        pattern_groups (dict[fx.Node, frozenset[fx.Node]]): For each
            covered node, the set of fx nodes belonging to the same
            winning pattern match.
        shared_observer_nodes (dict[fx.Node, type[SharedObserverModulePattern]]):
            Every shared-observer node in the model, mapped to the
            pattern class that owns its qspec-sharing semantics.
            :func:`_shared_observer_constraints` dispatches through
            this mapping.
    """

    winning_configs: dict[fx.Node, OpQuantizerConfig]
    node_priorities: dict[fx.Node, int]
    pattern_groups: dict[fx.Node, frozenset[fx.Node]]
    shared_observer_nodes: dict[fx.Node, type[SharedObserverModulePattern]]


# ---------------------------------------------------------------------------
# Entry point + drain loop.
# ---------------------------------------------------------------------------


def annotate_via_reconciliation(
    model: fx.GraphModule, ctx: _AnnotationContext
) -> fx.GraphModule:
    """Annotate ``model`` in place using the constraint-queue reconciler."""
    qspecs = build_initial_state(
        model,
        winning_configs=ctx.winning_configs,
        node_priorities=ctx.node_priorities,
        pattern_groups=ctx.pattern_groups,
    )

    queue: deque[Constraint] = deque()
    for node in model.graph.nodes:
        queue.extend(_generate_constraints_for_node(node, qspecs, ctx))

    # Runaway-loop backstop; convergence is guaranteed by priority
    # monotonicity but a bug in a policy could stall.
    max_iters = 1000 * (1 + sum(1 for _ in model.graph.nodes)) * 8
    iters = 0
    while queue:
        constraint = queue.popleft()
        changed = constraint.apply(qspecs)
        iters += 1
        if iters > max_iters:
            raise RuntimeError(
                f"Constraint drain did not converge after {iters} iterations — "
                f"likely a bug in a reconciliation policy or generator."
            )
        touched_nodes = {slot.node for slot in changed}
        for touched in touched_nodes:
            queue.extend(_generate_constraints_for_node(touched, qspecs, ctx))

    resolve_qspecs(model, qspecs)
    return model


# ---------------------------------------------------------------------------
# Per-node constraint dispatch.
# ---------------------------------------------------------------------------


def _generate_constraints_for_node(
    node: fx.Node, qspecs: ProvisionalQSpecMap, ctx: _AnnotationContext
) -> list[Constraint]:
    """Return every constraint whose scope includes ``node``.

    Rules run against the current ``qspecs`` (dynamic peeking allowed) so
    that when a slot on a downstream/upstream node has already been
    decided, rules can inspect the decision and emit stronger constraints.
    """
    constraints: list[Constraint] = []

    constraints.extend(_adjacent_edge_constraints(node, qspecs, ctx))

    pattern_class = ctx.shared_observer_nodes.get(node)
    if pattern_class is not None:
        constraints.extend(
            pattern_class.generate_qspec_sharing_constraints(node, qspecs)
        )

    if _is_state_node(node) and len(node.users) >= 2:
        constraints.extend(_shared_state_constraints(node, qspecs, ctx))

    return constraints


# ---------------------------------------------------------------------------
# Adjacent-edge sharing.
# ---------------------------------------------------------------------------


def _adjacent_edge_constraints(
    node: fx.Node, qspecs: ProvisionalQSpecMap, ctx: _AnnotationContext
) -> list[Constraint]:
    """For every fx edge touching ``node`` whose endpoints are both
    covered and non-internal, emit :class:`ShareObserverInstance` tying
    the producer's OUTPUT slot to the consumer's INPUT slot when at
    least one of the two has fields in ``qspecs``.

    Rationale mirrors the old code's :func:`_fill_input_qspec_map_for_input`:
    if a producer's output is observed and its consumer would also want
    to observe the same tensor, use one observer.

    Consequence — transitive grouping. Every constraint here is pairwise
    (producer.OUTPUT, consumer.INPUT[i]), but
    :meth:`ShareObserverInstance.apply` widens the group to include any
    slot already sharing a :class:`ProvisionalQSpec` with a member.
    Result: for a node ``A`` with fan-out to consumers ``B`` and ``C``,
    once (A→B) and (A→C) constraints both apply, ``A.OUTPUT``,
    ``B.INPUT[i]``, and ``C.INPUT[j]`` all end up in one observer group
    — one fake_quant module at runtime.

    **Forecloses the requant-at-edge option.** Torchao's annotation
    model natively supports having *different* concrete specs on
    ``producer.output_qspec`` and ``consumer.input_qspec_map[producer]``.
    At runtime that produces
    ``quant(producer_spec) → dequant → quant(consumer_spec) → dequant``
    — the "intentional requant boundary" pattern. Typical use case: a
    producer emits int8 activations while a specific consumer wants int4
    at that edge. By emitting :class:`ShareObserverInstance` on every
    adjacent edge, this function closes off that option: divergent-spec
    proposals lose to the reconciler's priority-based merge (with a
    ``DTYPE_OVERRIDDEN`` relaxation), never becoming two separate
    observers.

    If a future use case needs intentional requant boundaries, options
    include:

    * A per-edge or per-config "don't share" opt-out that suppresses
      the constraint for named edges.
    * Detecting same-priority dtype conflicts and choosing "leave the
      sides independent" instead of merging.
    * Flipping adjacent-edge sharing from opt-out to opt-in.

    Internal edges (both endpoints in the same pattern group) are
    skipped — pattern config declares specs for pattern boundary slots,
    not for edges between the pattern's own constituent nodes.
    """
    if node not in ctx.winning_configs:
        # Uncovered nodes (placeholders, unmatched ops) never anchor an
        # adjacent-edge constraint on either side; a covered neighbor
        # that cares about this edge emits it from its own side.
        return []

    constraints: list[Constraint] = []
    pattern = ctx.pattern_groups.get(node, frozenset({node}))

    # Incoming edges: (producer, node, arg_index).
    for arg_index, producer in enumerate(node.all_input_nodes):
        if producer in pattern:
            continue
        if producer not in ctx.winning_configs:
            continue
        producer_output = NodeSlot(node=producer, kind=SlotKind.OUTPUT, arg_index=0)
        consumer_input = NodeSlot(node=node, kind=SlotKind.INPUT, arg_index=arg_index)
        if producer_output in qspecs or consumer_input in qspecs:
            constraints.append(
                ShareObserverInstance(frozenset({producer_output, consumer_input}))
            )

    # Outgoing edges: (node, consumer, arg_index).
    for consumer in node.users:
        if consumer in pattern:
            continue
        if consumer not in ctx.winning_configs:
            continue
        consumer_input = _find_input_slot_for_producer(consumer, node)
        if consumer_input is None:
            continue
        producer_output = NodeSlot(node=node, kind=SlotKind.OUTPUT, arg_index=0)
        if producer_output in qspecs or consumer_input in qspecs:
            constraints.append(
                ShareObserverInstance(frozenset({producer_output, consumer_input}))
            )

    return constraints


def _find_input_slot_for_producer(
    consumer: fx.Node, producer: fx.Node
) -> NodeSlot | None:
    """Return the consumer's INPUT slot that reads from ``producer``, or
    ``None`` if no such slot exists (defensive; shouldn't happen for a
    well-formed users-relation).
    """
    for arg_index, actual_producer in enumerate(consumer.all_input_nodes):
        if actual_producer is producer:
            return NodeSlot(node=consumer, kind=SlotKind.INPUT, arg_index=arg_index)
    return None


# ---------------------------------------------------------------------------
# Shared-state (shared-weight) sharing.
# ---------------------------------------------------------------------------


def _shared_state_constraints(
    state_node: fx.Node, qspecs: ProvisionalQSpecMap, ctx: _AnnotationContext
) -> list[Constraint]:
    """Emit one :class:`ShareObserverInstance` tying every covered
    consumer's INPUT slot that reads ``state_node``.

    Weight tensors get quantized once at prepare time; all consumers of
    the same underlying tensor see the same quantized version at runtime.
    This constraint forces all covered consumers onto one qspec + one
    observer instance.

    **Forecloses divergent-spec consumers of a shared weight.** Torchao's
    annotation model natively supports two consumers of the same state
    tensor having different ``input_qspec_map[state]`` entries — it
    inserts one fake_quant per edge, each with its own spec. Both
    consumers receive their own quantized-then-dequantized view of the
    same underlying float tensor.

    **State-tensor-specific wrinkle.** Weight observer scale/zp is
    *derived* from the tensor values at prepare time (deterministic),
    not *learned* from calibration data at runtime. So
    :class:`ShareObserverInstance` on shared weights bundles two things
    that could be separated:

    1. Same qspec across consumers (dtype / qscheme / range agreement).
    2. Same observer instance (compute scale/zp once vs. per-consumer).

    For weights (2) is a trivial optimization — two observers on the
    same tensor compute identical values anyway. The strong constraint
    is (1). If a future use case needs divergent-spec consumers of a
    shared weight, loosening (1) is the right move — (2) falls out
    naturally.

    **Downstream storage/export tradeoff.** If divergent specs are
    allowed, ``convert_pt2e`` / export gets to choose the storage
    strategy: store multiple quantized versions (Nx storage, no runtime
    requant), or store the least aggressive version and requant to more
    aggressive variants at each edge (1x storage, one requant per
    divergent consumer). The reconciler doesn't need to pick — it
    just needs to permit divergent specs.
    """
    consumer_slots: list[NodeSlot] = []
    for consumer in state_node.users:
        if consumer not in ctx.winning_configs:
            continue
        consumer_input = _find_input_slot_for_producer(consumer, state_node)
        if consumer_input is None:
            continue
        consumer_slots.append(consumer_input)

    if len(consumer_slots) < 2:
        return []
    # Only fire when at least one consumer wants observation — if none
    # do, there's no observer to share.
    if not any(slot in qspecs for slot in consumer_slots):
        return []
    return [ShareObserverInstance(frozenset(consumer_slots))]


# ---------------------------------------------------------------------------
# Pattern coverage helper (used by quantizer.py when building the context).
# ---------------------------------------------------------------------------


def _nodes_covered_by(match_info: Any) -> list[fx.Node]:
    """Return every fx node an annotator match annotates."""
    match = match_info.annotator_match

    if isinstance(match, InternalMatch):
        return [
            node
            for key, node in match.name_node_map.items()
            if key in ("mod", "output", "bn")
        ]

    if isinstance(match, tuple) and all(
        isinstance(partition, SourcePartition) for partition in match
    ):
        return [_get_call_function_node_from_partition(partition) for partition in match]

    raise TypeError(
        f"Unknown annotator match type: {type(match).__name__}. Update "
        f"_nodes_covered_by when adding a new pattern family."
    )
