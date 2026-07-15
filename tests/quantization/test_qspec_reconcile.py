# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Unit tests for the constraint-queue qspec reconciliation pipeline.

Tests here are pure — they construct :class:`ProvisionalQSpec`
:class:`ProvisionalQSpecMap` values and :class:`Constraint`s by hand and
exercise ``.apply()`` directly. No fx graph construction. End-to-end
pipeline behavior is covered by trace-driven checks against the
walkthrough toy.
"""

from unittest.mock import Mock

import pytest
import torch
from torchao.quantization.pt2e.observer import MinMaxObserver

from coreai_opt.quantization._graph._qspec_constraints import (
    ShareFields,
    ShareObserverInstance,
    _reconcile_field,
)
from coreai_opt.quantization._graph._qspec_types import (
    FieldName,
    FieldValue,
    NodeSlot,
    ProvisionalQSpec,
    ProvisionalQSpecMap,
    ReconciliationError,
    SlotKind,
)

# ---------------------------------------------------------------------------
# Test helpers.
# ---------------------------------------------------------------------------


def _slot(name: str = "s", kind: SlotKind = SlotKind.OUTPUT, arg_index: int = 0) -> NodeSlot:
    """Opaque ``NodeSlot`` — reconciler never inspects the fx node."""
    return NodeSlot(node=Mock(name=name), kind=kind, arg_index=arg_index)


def _pspec(**fields: FieldValue) -> ProvisionalQSpec:
    """Build a ProvisionalQSpec by keyword: DTYPE=FieldValue(int8, 0), ..."""
    field_map = {FieldName[key]: value for key, value in fields.items()}
    return ProvisionalQSpec(fields=field_map)


def _fv(value, priority: int = 0) -> FieldValue:
    return FieldValue(value=value, priority=priority)


# ---------------------------------------------------------------------------
# _reconcile_field policies.
# ---------------------------------------------------------------------------


class TestReconcileField:
    def test_dtype_priority_wins(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(DTYPE=_fv(torch.int4, priority=1)),
            b: _pspec(DTYPE=_fv(torch.int8, priority=5)),
        }
        result = _reconcile_field(FieldName.DTYPE, frozenset({a, b}), state)
        assert result.value == torch.int4  # priority 1 beats 5
        assert result.priority == 1

    def test_dtype_tie_first_encountered(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(DTYPE=_fv(torch.int8, priority=3)),
            b: _pspec(DTYPE=_fv(torch.int4, priority=3)),
        }
        result = _reconcile_field(FieldName.DTYPE, frozenset({a, b}), state)
        # min() with equal keys returns the first — encounter order
        assert result.value in (torch.int8, torch.int4)

    def test_qscheme_lattice_symmetric_and_affine(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(QSCHEME=_fv(torch.per_tensor_symmetric, priority=0)),
            b: _pspec(QSCHEME=_fv(torch.per_tensor_affine, priority=5)),
        }
        result = _reconcile_field(FieldName.QSCHEME, frozenset({a, b}), state)
        assert result.value == torch.per_tensor_affine  # looser wins
        assert result.priority == 0  # min of {0, 5}

    def test_qscheme_lattice_per_tensor_and_per_channel(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(QSCHEME=_fv(torch.per_tensor_symmetric, priority=0)),
            b: _pspec(QSCHEME=_fv(torch.per_channel_symmetric, priority=0)),
        }
        result = _reconcile_field(FieldName.QSCHEME, frozenset({a, b}), state)
        assert result.value == torch.per_channel_symmetric

    def test_range_min_union(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(QUANT_MIN=_fv(-127, priority=0)),
            b: _pspec(QUANT_MIN=_fv(-128, priority=5)),
        }
        result = _reconcile_field(FieldName.QUANT_MIN, frozenset({a, b}), state)
        assert result.value == -128  # smaller wins for min
        assert result.priority == 0

    def test_range_max_union(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(QUANT_MAX=_fv(127, priority=0)),
            b: _pspec(QUANT_MAX=_fv(255, priority=5)),
        }
        result = _reconcile_field(FieldName.QUANT_MAX, frozenset({a, b}), state)
        assert result.value == 255  # larger wins for max
        assert result.priority == 0

    def test_ch_axis_must_agree(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(CH_AXIS=_fv(0, priority=0)),
            b: _pspec(CH_AXIS=_fv(1, priority=0)),
        }
        with pytest.raises(ReconciliationError):
            _reconcile_field(FieldName.CH_AXIS, frozenset({a, b}), state)

    def test_is_dynamic_must_agree(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(IS_DYNAMIC=_fv(True, priority=0)),
            b: _pspec(IS_DYNAMIC=_fv(False, priority=0)),
        }
        with pytest.raises(ReconciliationError):
            _reconcile_field(FieldName.IS_DYNAMIC, frozenset({a, b}), state)

    def test_missing_field_returns_none(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {a: _pspec(), b: _pspec()}
        assert _reconcile_field(FieldName.DTYPE, frozenset({a, b}), state) is None


# ---------------------------------------------------------------------------
# ShareFields.
# ---------------------------------------------------------------------------


class TestShareFields:
    def test_broadcasts_winner_to_all_slots(self) -> None:
        a, b, c = _slot("a"), _slot("b"), _slot("c")
        state: ProvisionalQSpecMap = {
            a: _pspec(DTYPE=_fv(torch.int4, priority=0)),
            b: _pspec(DTYPE=_fv(torch.int8, priority=5)),
            c: _pspec(),  # no proposal
        }
        con = ShareFields(_slots=frozenset({a, b, c}), fields=frozenset({FieldName.DTYPE}))
        changed = con.apply(state)
        assert changed == {b, c}  # a already had int4; b and c gain it
        assert state[a].fields[FieldName.DTYPE].value == torch.int4
        assert state[b].fields[FieldName.DTYPE].value == torch.int4
        assert state[c].fields[FieldName.DTYPE].value == torch.int4
        # All at priority 0 after reconciliation.
        assert state[a].fields[FieldName.DTYPE].priority == 0
        assert state[b].fields[FieldName.DTYPE].priority == 0

    def test_noop_when_already_reconciled(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(DTYPE=_fv(torch.int8, priority=0)),
            b: _pspec(DTYPE=_fv(torch.int8, priority=0)),
        }
        con = ShareFields(_slots=frozenset({a, b}), fields=frozenset({FieldName.DTYPE}))
        assert con.apply(state) == set()

    def test_multiple_fields(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(
                DTYPE=_fv(torch.int4, priority=0),
                QSCHEME=_fv(torch.per_tensor_symmetric, priority=0),
            ),
            b: _pspec(
                DTYPE=_fv(torch.int8, priority=5),
                QSCHEME=_fv(torch.per_tensor_affine, priority=5),
            ),
        }
        con = ShareFields(
            _slots=frozenset({a, b}),
            fields=frozenset({FieldName.DTYPE, FieldName.QSCHEME}),
        )
        con.apply(state)
        # a's dtype wins (higher priority); qscheme resolves via lattice.
        assert state[a].fields[FieldName.DTYPE].value == torch.int4
        assert state[b].fields[FieldName.DTYPE].value == torch.int4
        assert state[a].fields[FieldName.QSCHEME].value == torch.per_tensor_affine
        assert state[b].fields[FieldName.QSCHEME].value == torch.per_tensor_affine


# ---------------------------------------------------------------------------
# ShareObserverInstance.
# ---------------------------------------------------------------------------


class TestShareObserverInstance:
    def test_merges_two_slots_into_one_instance(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(DTYPE=_fv(torch.int8, priority=0)),
            b: _pspec(DTYPE=_fv(torch.int8, priority=0)),
        }
        assert state[a] is not state[b]
        con = ShareObserverInstance(_slots=frozenset({a, b}))
        con.apply(state)
        assert state[a] is state[b]  # identity, not just value equality

    def test_field_mutation_after_share_propagates(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {a: _pspec(), b: _pspec()}
        ShareObserverInstance(_slots=frozenset({a, b})).apply(state)
        state[a].fields[FieldName.DTYPE] = _fv(torch.int8, 0)
        # b's ProvisionalQSpec is the same object → sees the new field.
        assert state[b].fields[FieldName.DTYPE].value == torch.int8

    def test_transitive_merge_pulls_in_prior_sharers(self) -> None:
        a, b, c = _slot("a"), _slot("b"), _slot("c")
        state: ProvisionalQSpecMap = {a: _pspec(), b: _pspec(), c: _pspec()}
        # First merge a and b.
        ShareObserverInstance(_slots=frozenset({a, b})).apply(state)
        assert state[a] is state[b]
        # Now merge a with c — b should be pulled in transitively.
        ShareObserverInstance(_slots=frozenset({a, c})).apply(state)
        assert state[a] is state[b] is state[c]

    def test_reconciles_fields_across_merged_group(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(DTYPE=_fv(torch.int4, priority=0)),
            b: _pspec(DTYPE=_fv(torch.int8, priority=5)),
        }
        ShareObserverInstance(_slots=frozenset({a, b})).apply(state)
        assert state[a].fields[FieldName.DTYPE].value == torch.int4
        assert state[b].fields[FieldName.DTYPE].value == torch.int4
        assert state[a] is state[b]

    def test_noop_when_already_shared_with_correct_fields(self) -> None:
        a, b = _slot("a"), _slot("b")
        shared = _pspec(DTYPE=_fv(torch.int8, priority=0))
        state: ProvisionalQSpecMap = {a: shared, b: shared}
        changed = ShareObserverInstance(_slots=frozenset({a, b})).apply(state)
        assert changed == set()
        assert state[a] is shared
        assert state[b] is shared


# ---------------------------------------------------------------------------
# Convergence / no-op behavior.
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_repeated_share_fields_stabilizes(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(DTYPE=_fv(torch.int4, priority=1)),
            b: _pspec(DTYPE=_fv(torch.int8, priority=5)),
        }
        con = ShareFields(_slots=frozenset({a, b}), fields=frozenset({FieldName.DTYPE}))
        first = con.apply(state)
        assert first  # something changed
        # Immediate re-apply is a no-op — proves the priority-inheritance
        # rule prevents oscillation.
        assert con.apply(state) == set()

    def test_repeated_share_observer_stabilizes(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {a: _pspec(), b: _pspec()}
        con = ShareObserverInstance(_slots=frozenset({a, b}))
        first = con.apply(state)
        assert first
        assert con.apply(state) == set()


# ---------------------------------------------------------------------------
# YOLOX-shape scenario reconciliation.
# ---------------------------------------------------------------------------


def test_yolox_scenario_end_to_end_reconciliation() -> None:
    """cat-of-sigmoid: two sigmoid-forced-affine outputs plus a wide-range
    symmetric conv output. Group unifies to affine (lattice join), dtype
    picks the highest-priority proposal."""
    conv_out = _slot("conv_out")
    sig_a_out = _slot("sig_a_out")
    sig_b_out = _slot("sig_b_out")
    cat_out = _slot("cat_out")

    state: ProvisionalQSpecMap = {
        conv_out: _pspec(
            DTYPE=_fv(torch.int8, priority=5),
            QSCHEME=_fv(torch.per_tensor_symmetric, priority=5),
            QUANT_MIN=_fv(-128, priority=5),
            QUANT_MAX=_fv(127, priority=5),
            OBSERVER_CLASS=_fv(MinMaxObserver, priority=5),
            IS_DYNAMIC=_fv(False, priority=5),
        ),
        sig_a_out: _pspec(
            DTYPE=_fv(torch.int8, priority=5),
            QSCHEME=_fv(torch.per_tensor_affine, priority=5),  # sigmoid intrinsic
            QUANT_MIN=_fv(-128, priority=5),
            QUANT_MAX=_fv(127, priority=5),
            OBSERVER_CLASS=_fv(MinMaxObserver, priority=5),
            IS_DYNAMIC=_fv(False, priority=5),
        ),
        sig_b_out: _pspec(
            DTYPE=_fv(torch.int8, priority=5),
            QSCHEME=_fv(torch.per_tensor_affine, priority=5),
            QUANT_MIN=_fv(-128, priority=5),
            QUANT_MAX=_fv(127, priority=5),
            OBSERVER_CLASS=_fv(MinMaxObserver, priority=5),
            IS_DYNAMIC=_fv(False, priority=5),
        ),
        cat_out: _pspec(
            DTYPE=_fv(torch.int8, priority=5),
            QSCHEME=_fv(torch.per_tensor_symmetric, priority=5),
            QUANT_MIN=_fv(-128, priority=5),
            QUANT_MAX=_fv(127, priority=5),
            OBSERVER_CLASS=_fv(MinMaxObserver, priority=5),
            IS_DYNAMIC=_fv(False, priority=5),
        ),
    }
    ShareObserverInstance(_slots=frozenset({conv_out, sig_a_out, sig_b_out, cat_out})).apply(state)
    # All four share one ProvisionalQSpec now.
    assert state[conv_out] is state[sig_a_out] is state[sig_b_out] is state[cat_out]
    # Reconciled qscheme is affine (looser wins).
    assert state[conv_out].fields[FieldName.QSCHEME].value == torch.per_tensor_affine
    assert state[conv_out].fields[FieldName.DTYPE].value == torch.int8


# ---------------------------------------------------------------------------
# Adjacent-edge sharing — the peephole case from precedence tests.
# ---------------------------------------------------------------------------


def test_adjacent_edge_share_resolves_dtype_conflict_by_priority() -> None:
    """test_op_level_precedence-shape: linear1.OUTPUT wants int4 at lower
    priority; linear2.INPUT wants int8 at higher priority. Adjacent-edge
    sharing merges them; priority resolves dtype to int8."""
    linear1_out = _slot("linear1_out", kind=SlotKind.OUTPUT)
    linear2_in = _slot("linear2_in", kind=SlotKind.INPUT)
    state: ProvisionalQSpecMap = {
        linear1_out: _pspec(
            DTYPE=_fv(torch.int4, priority=5),
            QSCHEME=_fv(torch.per_tensor_symmetric, priority=5),
            OBSERVER_CLASS=_fv(MinMaxObserver, priority=5),
            IS_DYNAMIC=_fv(False, priority=5),
        ),
        linear2_in: _pspec(
            DTYPE=_fv(torch.int8, priority=0),
            QSCHEME=_fv(torch.per_tensor_symmetric, priority=0),
            OBSERVER_CLASS=_fv(MinMaxObserver, priority=0),
            IS_DYNAMIC=_fv(False, priority=0),
        ),
    }
    ShareObserverInstance(_slots=frozenset({linear1_out, linear2_in})).apply(state)
    assert state[linear1_out] is state[linear2_in]
    assert state[linear1_out].fields[FieldName.DTYPE].value == torch.int8


# ---------------------------------------------------------------------------
# Concat axis-aware sharing — per-channel on concat axis stays independent.
# ---------------------------------------------------------------------------


def test_share_fields_dtype_only_leaves_scale_independent() -> None:
    """When per-channel is along the concat axis, only DTYPE is shared —
    scale/zp (via observer instance) stays per-input."""
    a_out = _slot("a_out")
    b_out = _slot("b_out")
    state: ProvisionalQSpecMap = {
        a_out: _pspec(DTYPE=_fv(torch.int8, priority=0)),
        b_out: _pspec(DTYPE=_fv(torch.int4, priority=5)),
    }
    ShareFields(_slots=frozenset({a_out, b_out}), fields=frozenset({FieldName.DTYPE})).apply(state)
    # dtypes agree, but the ProvisionalQSpec objects are still distinct.
    assert state[a_out] is not state[b_out]
    assert state[a_out].fields[FieldName.DTYPE].value == torch.int8
    assert state[b_out].fields[FieldName.DTYPE].value == torch.int8
