# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Phase 4 of the qspec reconciliation pipeline: resolve the final
:class:`ProvisionalQSpecMap` onto per-node
:class:`QuantizationAnnotation` fields on the fx graph.

By the time this runs, the constraint queue has drained, every slot
has its reconciled fields, and slots that must share an observer at
runtime are already pointing at the same :class:`ProvisionalQSpec`
object. This module's job is to project that state onto torchao's
per-node data model:

    1. Group slots by :class:`ProvisionalQSpec` object identity.
    2. For each group, pick an anchor slot (topologically-first;
       INPUT preferred over OUTPUT within a node — see
       :class:`SlotOrderKey` for why).
    3. Assign the anchor a concrete
       :class:`torchao...QuantizationSpec`; the others a
       :class:`torchao...SharedQuantizationSpec` pointing at the anchor.
    4. Bucket per node, backfill missing input slots with ``None``
       (a torchao interop concession), and write
       :class:`QuantizationAnnotation` entries.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch.fx as fx
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec as TorchAOQuantizationSpec,
    SharedQuantizationSpec as _SharedQuantizationSpec,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

from ._qspec_types import (
    FieldName,
    NodeSlot,
    ProvisionalQSpec,
    ProvisionalQSpecMap,
    SlotKind,
)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def resolve_qspecs(model: fx.GraphModule, qspecs: ProvisionalQSpecMap) -> None:
    """Project reconciled per-slot state onto per-node annotations.

    Mutates ``model``'s node metadata in place. See module docstring for
    the four-step pipeline.
    """
    groups = _group_slots_by_qspec_identity(qspecs)
    topo_index = _topo_index(model)

    slot_assignments: dict[NodeSlot, Any] = {}
    for group in groups:
        _assign_group_specs(group, topo_index, slot_assignments)

    per_node_inputs, per_node_outputs = _bucket_per_node(slot_assignments)
    _write_annotations(model, per_node_inputs, per_node_outputs)


# ---------------------------------------------------------------------------
# Grouping.
# ---------------------------------------------------------------------------


@dataclass
class _QSpecGroup:
    """One equivalence class of slots — every slot points at the same
    :class:`ProvisionalQSpec` object.
    """

    qspec: ProvisionalQSpec
    slots: list[NodeSlot]


def _group_slots_by_qspec_identity(qspecs: ProvisionalQSpecMap) -> list[_QSpecGroup]:
    """Bucket slots by ``id(ProvisionalQSpec)`` — same object = same group."""
    groups_by_id: dict[int, _QSpecGroup] = {}
    for slot, qspec in qspecs.items():
        group = groups_by_id.get(id(qspec))
        if group is None:
            group = _QSpecGroup(qspec=qspec, slots=[])
            groups_by_id[id(qspec)] = group
        group.slots.append(slot)
    return list(groups_by_id.values())


# ---------------------------------------------------------------------------
# Anchor selection.
# ---------------------------------------------------------------------------


@dataclass(order=True, frozen=True)
class SlotOrderKey:
    """Sort key for anchor selection within a shared-observer group.

    Dataclasses with ``order=True`` compare field-by-field in declaration
    order, so the ordering below is exactly ``(topo_index, kind_order,
    arg_index)`` lexicographically.

    Attributes:
        topo_index (int): Position of the slot's node in
            ``model.graph.nodes`` iteration order. Torchao processes
            nodes in this order; the anchor must be topo-first so its
            observer is registered in ``obs_or_fq_map`` before any
            downstream SharedSpec references it.
        kind_order (int): Within a single node, INPUT (0) sorts before
            OUTPUT (1). Torchao's ``_get_edge_or_node_to_qspec``
            iterates ``input_qspec_map`` entries before ``output_qspec``
            for a given node; anchoring on the input side ensures the
            observer is registered by the time the OUTPUT-side SharedSpec
            on the same node resolves.
        arg_index (int): For INPUT slots, distinguishes positional
            inputs. Tiebreaker only; the anchor's position doesn't
            affect torchao's semantics.
    """

    topo_index: int
    kind_order: int
    arg_index: int


def _topo_index(model: fx.GraphModule) -> dict[fx.Node, int]:
    return {node: i for i, node in enumerate(model.graph.nodes)}


def _slot_order_key(slot: NodeSlot, topo_index: dict[fx.Node, int]) -> SlotOrderKey:
    return SlotOrderKey(
        topo_index=topo_index.get(slot.node, len(topo_index)),
        kind_order=0 if slot.kind is SlotKind.INPUT else 1,
        arg_index=slot.arg_index,
    )


def _pick_anchor(group: _QSpecGroup, topo_index: dict[fx.Node, int]) -> NodeSlot:
    """Return the topologically-first slot in ``group``. See
    :class:`SlotOrderKey` for why INPUT is preferred over OUTPUT within
    a single node."""
    return min(group.slots, key=lambda slot: _slot_order_key(slot, topo_index))


# ---------------------------------------------------------------------------
# Spec assignment.
# ---------------------------------------------------------------------------


def _assign_group_specs(
    group: _QSpecGroup,
    topo_index: dict[fx.Node, int],
    slot_assignments: dict[NodeSlot, Any],
) -> None:
    """Fill ``slot_assignments`` with the spec each slot in ``group`` should carry.

    Singleton groups (or all-``None`` groups) just get the concrete spec
    on the one slot. Multi-slot groups pick an anchor (concrete spec)
    and give the rest a :class:`SharedQuantizationSpec` pointing back.
    """
    concrete = _build_concrete_spec(group.qspec)
    if concrete is None:
        # Empty or None-only group — no annotation to emit.
        return
    if len(group.slots) == 1:
        slot_assignments[group.slots[0]] = concrete
        return

    anchor = _pick_anchor(group, topo_index)
    shared_ref = _shared_spec_pointing_at(anchor)
    for slot in group.slots:
        slot_assignments[slot] = concrete if slot == anchor else shared_ref


def _shared_spec_pointing_at(anchor: NodeSlot) -> _SharedQuantizationSpec:
    """Build a :class:`SharedQuantizationSpec` in the form matching where
    torchao registers ``anchor``'s observer.

    Torchao's ``obs_or_fq_map`` uses different keys depending on which
    annotation field placed the observer:

    * ``node.output_qspec = spec`` → registered under key ``node``.
      Reference via **Node form** ``SharedQuantizationSpec(node)``.
    * ``consumer.input_qspec_map[producer] = spec`` → registered under
      key ``(producer, consumer)``. Reference via **edge form**
      ``SharedQuantizationSpec((producer, consumer))``.
    """
    if anchor.kind is SlotKind.OUTPUT:
        return _SharedQuantizationSpec(anchor.node)
    producer = anchor.node.all_input_nodes[anchor.arg_index]
    return _SharedQuantizationSpec((producer, anchor.node))


def _build_concrete_spec(qspec: ProvisionalQSpec) -> TorchAOQuantizationSpec | None:
    """Assemble a :class:`TorchAOQuantizationSpec` from reconciled fields.

    Returns ``None`` if required fields (``DTYPE``, ``OBSERVER_CLASS``) are
    missing — that ProvisionalQSpec represents a slot the pipeline chose
    not to observe (e.g. a bias slot with ``op_state_spec`` unset).
    """
    if FieldName.DTYPE not in qspec.fields or FieldName.OBSERVER_CLASS not in qspec.fields:
        return None
    return TorchAOQuantizationSpec(
        dtype=qspec.fields[FieldName.DTYPE].value,
        observer_or_fake_quant_ctr=qspec.fields[FieldName.OBSERVER_CLASS].value,
        quant_min=(
            qspec.fields[FieldName.QUANT_MIN].value
            if FieldName.QUANT_MIN in qspec.fields
            else None
        ),
        quant_max=(
            qspec.fields[FieldName.QUANT_MAX].value
            if FieldName.QUANT_MAX in qspec.fields
            else None
        ),
        qscheme=(
            qspec.fields[FieldName.QSCHEME].value
            if FieldName.QSCHEME in qspec.fields
            else None
        ),
        ch_axis=(
            qspec.fields[FieldName.CH_AXIS].value
            if FieldName.CH_AXIS in qspec.fields
            else None
        ),
        is_dynamic=(
            qspec.fields[FieldName.IS_DYNAMIC].value
            if FieldName.IS_DYNAMIC in qspec.fields
            else False
        ),
    )


# ---------------------------------------------------------------------------
# Per-node bucketing + write.
# ---------------------------------------------------------------------------


def _bucket_per_node(
    slot_assignments: dict[NodeSlot, Any],
) -> tuple[dict[fx.Node, dict[fx.Node, Any]], dict[fx.Node, Any]]:
    """Split per-slot spec assignments into per-node ``input_qspec_map``
    and per-node ``output_qspec`` buckets.
    """
    per_node_inputs: dict[fx.Node, dict[fx.Node, Any]] = defaultdict(dict)
    per_node_outputs: dict[fx.Node, Any] = {}
    for slot, spec in slot_assignments.items():
        if slot.kind is SlotKind.OUTPUT:
            per_node_outputs[slot.node] = spec
        else:
            producer = slot.node.all_input_nodes[slot.arg_index]
            per_node_inputs[slot.node][producer] = spec
    return per_node_inputs, per_node_outputs


def _write_annotations(
    model: fx.GraphModule,
    per_node_inputs: dict[fx.Node, dict[fx.Node, Any]],
    per_node_outputs: dict[fx.Node, Any],
) -> None:
    """Mutate each touched node's meta with a :class:`QuantizationAnnotation`.

    Nodes with any input-side annotation get their ``input_qspec_map``
    backfilled with ``None`` for every positional input that didn't get
    an explicit spec — torchao's downstream passes (notably
    ``qat_utils.py``'s ``_replace_target_node_with_quantization_annotation``)
    look up slots by positional index and IndexError if entries are
    missing. This is a torchao interop concession, not a reconciliation
    decision.
    """
    touched: set[fx.Node] = set(per_node_inputs) | set(per_node_outputs)
    for node in touched:
        annotation = node.meta.get(Q_ANNOTATION_KEY, QuantizationAnnotation())
        input_map = per_node_inputs.get(node)
        if input_map:
            annotation.input_qspec_map = _backfill_input_qspec_map(node, input_map)
        if node in per_node_outputs:
            annotation.output_qspec = per_node_outputs[node]
        annotation._annotated = True
        node.meta[Q_ANNOTATION_KEY] = annotation


def _backfill_input_qspec_map(
    node: fx.Node, input_map: dict[fx.Node, Any]
) -> dict[fx.Node, Any]:
    """Return an ``input_qspec_map`` with an entry for every
    ``node.all_input_nodes`` position — explicit specs where set,
    ``None`` otherwise.
    """
    return {
        producer: input_map.get(producer, None)
        for producer in node.all_input_nodes
    }
