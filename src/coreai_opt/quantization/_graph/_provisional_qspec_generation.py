# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Phase 2 of the qspec reconciliation pipeline: build the initial
:class:`ProvisionalQSpecMap` from winning configs and op-intrinsic
overrides.

For each pattern-covered node the pipeline seeds field values on:

    * Each INPUT slot whose producer is outside the node's pattern
      group (state producers get ``op_state_spec``; everything else
      gets ``op_input_spec``).
    * The OUTPUT slot, unless every consumer is inside the pattern
      group (internal edges get no annotation — see below).

For nodes whose target has op-intrinsic quantization semantics
(sigmoid/tanh/hardsigmoid/relu/relu6), the intrinsic overrides the
user's proposal on the corresponding fields. Every override that
actually changes a user-specified value is logged.

Two pattern-shape assumptions are checked and raised on violation:

    * An op-intrinsic node whose OUTPUT slot would be skipped (all its
      consumers live inside its pattern group) means the intrinsic is
      about to be silently dropped. Raise instead — a future pattern
      author must acknowledge the conflict.
    * A pattern-covered node with a mix of in-pattern and external
      consumers can't have the pattern's "no observer on internal
      edges" contract honored (the observer torchao inserts at the
      producer's OUTPUT is visible to both consumer groups). Raise.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.fx as fx
from torchao.quantization.pt2e.fake_quantize import FixedQParamsFakeQuantize
from torchao.quantization.pt2e.quantizer import QuantizationSpec as TorchAOQuantizationSpec

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
)
from ._qspec_constraints import _get_or_create
from ._qspec_types import (
    FieldName,
    FieldValue,
    NodeSlot,
    ProvisionalQSpecMap,
    ReconciliationError,
    SlotKind,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def build_initial_state(
    model: fx.GraphModule,
    winning_configs: dict[fx.Node, OpQuantizerConfig],
    node_priorities: dict[fx.Node, int],
    pattern_groups: dict[fx.Node, frozenset[fx.Node]],
) -> ProvisionalQSpecMap:
    """Build the initial :class:`ProvisionalQSpecMap` from
    winning-config + intrinsic contributions.
    """
    node_to_annotation_config: dict[fx.Node, AnnotationConfig] = {
        node: AnnotationConfig.from_quantizer_config(cfg)
        for node, cfg in winning_configs.items()
    }

    qspecs: ProvisionalQSpecMap = {}
    for node, cfg in node_to_annotation_config.items():
        priority = node_priorities[node]
        pattern = pattern_groups.get(node, frozenset({node}))
        _populate_input_slots(node, cfg, priority, pattern, qspecs)
        _populate_output_slot(node, cfg, priority, pattern, qspecs)

    return qspecs


# ---------------------------------------------------------------------------
# Input-slot population.
# ---------------------------------------------------------------------------


def _populate_input_slots(
    node: fx.Node,
    cfg: AnnotationConfig,
    priority: int,
    pattern: frozenset[fx.Node],
    qspecs: ProvisionalQSpecMap,
) -> None:
    """Populate every non-internal INPUT slot on ``node`` from its config."""
    for arg_index, producer in enumerate(node.all_input_nodes):
        if producer in pattern:
            continue  # internal edge — skip
        slot = NodeSlot(node=node, kind=SlotKind.INPUT, arg_index=arg_index)
        if _is_state_node(producer):
            _validate_state_not_referenced_via_input_spec(producer, arg_index, cfg)
            spec = _lookup_state_spec(cfg, producer)
        else:
            spec = _lookup_by_key(cfg.op_input_spec, arg_index)
        if spec is None:
            continue
        _populate_fields_from_spec(qspecs, slot, spec, priority)


def _validate_state_not_referenced_via_input_spec(
    state_producer: fx.Node, arg_index: int, cfg: AnnotationConfig
) -> None:
    """Raise if the user's ``op_input_spec`` targets an arg index whose
    producer is a state tensor. State tensors must be configured via
    ``op_state_spec``.
    """
    if arg_index in cfg.op_input_spec:
        raise RuntimeError(
            f"Config is attempting to set op_input_spec idx {arg_index}, "
            f"but the input is a state tensor (node: {state_producer.name}). "
            f"Use op_state_spec to configure state inputs instead.\n"
            f"op_input_spec: {cfg.op_input_spec}"
        )


def _lookup_state_spec(
    consumer_cfg: AnnotationConfig, state_node: fx.Node
) -> TorchAOQuantizationSpec | None:
    """Resolve a state consumer's spec from its ``op_state_spec``."""
    state_name = _get_local_state_name(state_node)
    if state_name is None:
        return None
    return _lookup_by_key(consumer_cfg.op_state_spec, state_name)


# ---------------------------------------------------------------------------
# Output-slot population + op-intrinsic override.
# ---------------------------------------------------------------------------


def _populate_output_slot(
    node: fx.Node,
    cfg: AnnotationConfig,
    priority: int,
    pattern: frozenset[fx.Node],
    qspecs: ProvisionalQSpecMap,
) -> None:
    """Populate ``node``'s OUTPUT slot from its config, then apply any
    op-intrinsic override.

    Enforces two pattern-shape invariants:

    * Op-intrinsic nodes must not be fully internal to a pattern (else
      their intrinsic contribution is silently dropped — an unhandled
      case for any future pattern that places an intrinsic op mid-chain).
    * A covered node must have its consumers either all inside the
      pattern or all outside (any mix means the pattern's implicit "no
      observer on the internal edge" contract can't be honored).
    """
    if not node.users:
        return

    consumers_in_pattern = [
        consumer for consumer in node.users if consumer in pattern
    ]
    consumers_outside_pattern = [
        consumer for consumer in node.users if consumer not in pattern
    ]

    has_intrinsic = _op_intrinsic_qscheme(node) is not None

    if consumers_in_pattern and consumers_outside_pattern:
        raise ReconciliationError(
            f"Pattern-covered node {node.name!r} (target={node.target!r}) has "
            f"consumers both inside and outside its pattern group. The "
            f"pattern's 'no observer on internal edges' convention can't "
            f"be honored — an observer inserted at {node.name}.output for "
            f"the external consumer(s) would also be visible to the "
            f"in-pattern consumer(s).\n"
            f"  pattern group: {sorted(covered.name for covered in pattern)}\n"
            f"  in-pattern consumers: {sorted(consumer.name for consumer in consumers_in_pattern)}\n"
            f"  external consumers: {sorted(consumer.name for consumer in consumers_outside_pattern)}\n"
            f"If this pattern is intentional, update "
            f"_provisional_qspec_generation to decide how the internal "
            f"and external consumers should share (or not share) the "
            f"same observer."
        )

    fully_internal = not consumers_outside_pattern
    if fully_internal:
        if has_intrinsic:
            raise ReconciliationError(
                f"Op-intrinsic node {node.name!r} (target={node.target!r}) "
                f"has all consumers inside its pattern group "
                f"{sorted(covered.name for covered in pattern)}. Its "
                f"intrinsic qspec (qscheme / range / observer) would be "
                f"silently dropped because internal-edge OUTPUT slots "
                f"aren't annotated.\n"
                f"If a new pattern legitimately places an intrinsic op "
                f"mid-chain, update _provisional_qspec_generation to "
                f"decide how the intrinsic's constraints should be applied "
                f"(likely by carrying them onto some other slot in the "
                f"pattern boundary)."
            )
        return

    output_slot = NodeSlot(node=node, kind=SlotKind.OUTPUT, arg_index=0)
    out_spec = _lookup_by_key(cfg.op_output_spec, 0)
    if out_spec is None:
        return

    _populate_fields_from_spec(qspecs, output_slot, out_spec, priority)
    if has_intrinsic:
        _apply_op_intrinsic_override(node, output_slot, out_spec, priority, qspecs)


def _apply_op_intrinsic_override(
    node: fx.Node,
    output_slot: NodeSlot,
    user_spec: TorchAOQuantizationSpec,
    priority: int,
    qspecs: ProvisionalQSpecMap,
) -> None:
    """Overwrite the user's OUTPUT-slot fields with the op-intrinsic's known values.

    For :data:`_fixed_q_params_ops` nodes (sigmoid / tanh / hardsigmoid),
    the intrinsic carries a full :class:`FixedQParamsQuantizationSpec` with
    known ``qscheme``, ``quant_min``, ``quant_max``, ``scale``, and
    ``zero_point``. If the user's dtype matches the intrinsic's, we
    enforce all of them (range fields written directly; scale/zp routed
    through ``OBSERVER_CLASS`` via a ``FixedQParamsFakeQuantize.with_args``
    partial). If the dtype differs, we only enforce ``QSCHEME`` — the
    range/scale/zp values are dtype-specific and don't translate.

    For :data:`_always_affine_ops` (relu / relu6 / hardtanh(0,6)), only
    ``QSCHEME`` (per_tensor_affine) is known; that's all we override.

    Every override that changes a value the user explicitly set is logged.
    """
    fixed_spec = _fixed_q_params_ops.get(node.target)
    override_qscheme = _op_intrinsic_qscheme(node)
    assert override_qscheme is not None, "caller must gate on _op_intrinsic_qscheme"

    _override_field(
        qspecs, output_slot, FieldName.QSCHEME,
        override_qscheme, priority, node, user_spec.qscheme, "qscheme",
    )

    if fixed_spec is None:
        # always-affine op — only qscheme is known.
        return

    dtype_matches = user_spec.dtype == fixed_spec.dtype
    if not dtype_matches:
        logger.warning(
            "Op-intrinsic override on %s (target=%s): user dtype %s doesn't "
            "match intrinsic dtype %s. Enforcing qscheme only; skipping "
            "range/scale/zero_point overrides because they're dtype-specific.",
            node.name, node.target, user_spec.dtype, fixed_spec.dtype,
        )
        return

    _override_field(
        qspecs, output_slot, FieldName.QUANT_MIN,
        fixed_spec.quant_min, priority, node, user_spec.quant_min, "quant_min",
    )
    _override_field(
        qspecs, output_slot, FieldName.QUANT_MAX,
        fixed_spec.quant_max, priority, node, user_spec.quant_max, "quant_max",
    )
    # scale/zp are encoded by swapping the observer class for a partial that
    # pre-bakes the fixed parameters — a regular QuantizationSpec has no
    # scale/zp fields.
    fixed_observer_partial = FixedQParamsFakeQuantize.with_args(
        scale=fixed_spec.scale,
        zero_point=fixed_spec.zero_point,
        quant_min=fixed_spec.quant_min,
        quant_max=fixed_spec.quant_max,
        dtype=fixed_spec.dtype,
        qscheme=fixed_spec.qscheme,
    )
    _override_field(
        qspecs, output_slot, FieldName.OBSERVER_CLASS,
        fixed_observer_partial, priority, node,
        user_spec.observer_or_fake_quant_ctr, "observer_class",
    )


def _override_field(
    qspecs: ProvisionalQSpecMap,
    slot: NodeSlot,
    field_name: FieldName,
    intrinsic_value: Any,
    priority: int,
    node: fx.Node,
    user_value: Any,
    human_field_name: str,
) -> None:
    """Write ``intrinsic_value`` at ``field_name`` on ``slot``, logging if
    the user requested a different value.

    ``user_value`` is the value the user's spec carried (before the
    override), and is compared to ``intrinsic_value`` to decide whether
    the override is worth surfacing. Silent when they already agree or
    the user didn't specify.
    """
    qspec = _get_or_create(qspecs, slot)
    qspec.fields[field_name] = FieldValue(value=intrinsic_value, priority=priority)
    if user_value is not None and user_value != intrinsic_value:
        logger.info(
            "Op-intrinsic override on %s (target=%s): user %s=%r overridden "
            "by intrinsic %s=%r.",
            node.name, node.target, human_field_name, user_value,
            human_field_name, intrinsic_value,
        )


def _op_intrinsic_qscheme(node: fx.Node) -> Any | None:
    """Return the qscheme forced by op semantics, or ``None``."""
    if node.target in _fixed_q_params_ops:
        return _fixed_q_params_ops[node.target].qscheme
    if node.target in _always_affine_ops:
        return torch.per_tensor_affine
    if node.target in (torch.ops.aten.hardtanh.default, torch.ops.aten.hardtanh_.default):
        if len(node.args) >= 3 and node.args[1] == 0 and node.args[2] == 6:
            return torch.per_tensor_affine
    return None


# ---------------------------------------------------------------------------
# Spec-population helpers.
# ---------------------------------------------------------------------------


def _populate_fields_from_spec(
    qspecs: ProvisionalQSpecMap,
    slot: NodeSlot,
    spec: TorchAOQuantizationSpec,
    priority: int,
) -> None:
    """Copy every present field from a torchao :class:`QuantizationSpec` into ``qspecs``."""
    qspec = _get_or_create(qspecs, slot)
    qspec.fields[FieldName.DTYPE] = FieldValue(value=spec.dtype, priority=priority)
    qspec.fields[FieldName.OBSERVER_CLASS] = FieldValue(
        value=spec.observer_or_fake_quant_ctr, priority=priority
    )
    qspec.fields[FieldName.IS_DYNAMIC] = FieldValue(value=spec.is_dynamic, priority=priority)
    if spec.quant_min is not None:
        qspec.fields[FieldName.QUANT_MIN] = FieldValue(value=spec.quant_min, priority=priority)
    if spec.quant_max is not None:
        qspec.fields[FieldName.QUANT_MAX] = FieldValue(value=spec.quant_max, priority=priority)
    if spec.qscheme is not None:
        qspec.fields[FieldName.QSCHEME] = FieldValue(value=spec.qscheme, priority=priority)
    if spec.ch_axis is not None:
        qspec.fields[FieldName.CH_AXIS] = FieldValue(value=spec.ch_axis, priority=priority)


def _lookup_by_key(spec_map: dict[Any, Any], key: Any) -> Any:
    """Look up ``key`` in an ``op_input/op_output/op_state`` map with ``*`` fallback."""
    if key in spec_map:
        return spec_map[key]
    if _ALL_TENSORS in spec_map:
        return spec_map[_ALL_TENSORS]
    return None
