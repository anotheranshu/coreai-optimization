# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Constraint types + per-field reconciliation policies.

The two :class:`Constraint` subclasses (:class:`ShareFields`,
:class:`ShareObserverInstance`) are the vocabulary constraint generators
emit and the drain loop applies. Each per-field reconciliation rule
lives here as a small pure function, dispatched via
:data:`_FIELD_POLICY` in :func:`_reconcile_field`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torchao.quantization.pt2e.fake_quantize import FixedQParamsFakeQuantize
from torchao.quantization.pt2e.observer import FixedQParamsObserver

from ._qspec_types import (
    FieldName,
    FieldValue,
    NodeSlot,
    ProvisionalQSpec,
    ProvisionalQSpecMap,
    ReconciliationError,
)

logger = logging.getLogger(__name__)


_ALL_FIELDS: frozenset[FieldName] = frozenset(FieldName)


# ---------------------------------------------------------------------------
# Constraint ABC + implementations.
# ---------------------------------------------------------------------------


class Constraint(ABC):
    """A relation over one or more slots.

    :meth:`apply` reads the current :class:`ProvisionalQSpecMap`,
    mutates it to enforce the constraint, and returns the set of slots
    whose state changed. Returning an empty set means the constraint was
    already satisfied — that's the fixed-point signal the drain loop
    uses to detect quiescence.
    """

    @property
    @abstractmethod
    def slots(self) -> frozenset[NodeSlot]: ...

    @abstractmethod
    def apply(self, qspecs: ProvisionalQSpecMap) -> set[NodeSlot]: ...


@dataclass(frozen=True)
class ShareFields(Constraint):
    """Every slot in ``slots`` must agree on each of ``fields``.

    Reconciliation is per-field (see :func:`_reconcile_field`). The
    winning value is broadcast to every slot in the group; slots that
    had no proposal for a field pick up the winner. Priority of the
    reconciled value inherits the highest priority (lowest priority-
    number) among the contributing proposals — a monotonicity that
    guarantees the drain terminates.
    """

    _slots: frozenset[NodeSlot]
    fields: frozenset[FieldName]

    @property
    def slots(self) -> frozenset[NodeSlot]:
        return self._slots

    def apply(self, qspecs: ProvisionalQSpecMap) -> set[NodeSlot]:
        changed: set[NodeSlot] = set()
        for field_name in self.fields:
            reconciled = _reconcile_field(field_name, self._slots, qspecs)
            if reconciled is None:
                continue
            for slot in self._slots:
                qspec = _get_or_create(qspecs, slot)
                current = qspec.fields.get(field_name)
                if _field_value_stronger(reconciled, current):
                    qspec.fields[field_name] = reconciled
                    changed.add(slot)
        return changed


@dataclass(frozen=True)
class ShareObserverInstance(Constraint):
    """Every slot in ``slots`` must reference the same
    :class:`ProvisionalQSpec` object (same observer instance at runtime).

    Applying:

        1. Widen the group to include every slot already sharing a
           :class:`ProvisionalQSpec` with any member of ``self.slots``.
        2. Reconcile every field across the widened group via the same
           per-field policies :class:`ShareFields` uses.
        3. Re-point every widened slot's map entry at one shared
           :class:`ProvisionalQSpec`.

    After apply, mutating any member's fields propagates to every
    sharer — object identity IS the sharing relation.
    """

    _slots: frozenset[NodeSlot]

    @property
    def slots(self) -> frozenset[NodeSlot]:
        return self._slots

    def apply(self, qspecs: ProvisionalQSpecMap) -> set[NodeSlot]:
        # Widen: pull in every slot that already shares a
        # ProvisionalQSpec with any input slot.
        target_ids = {id(qspecs[slot]) for slot in self._slots if slot in qspecs}
        widened: set[NodeSlot] = set(self._slots)
        for slot, qspec in qspecs.items():
            if id(qspec) in target_ids:
                widened.add(slot)

        # Reconcile every field across the widened group.
        reconciled_fields: dict[FieldName, FieldValue] = {}
        for field_name in _ALL_FIELDS:
            reconciled = _reconcile_field(field_name, frozenset(widened), qspecs)
            if reconciled is not None:
                reconciled_fields[field_name] = reconciled

        changed: set[NodeSlot] = set()
        first_ref: ProvisionalQSpec | None = None
        for slot in widened:
            existing = qspecs.get(slot)
            already_reconciled = existing is not None and existing.fields == reconciled_fields
            if first_ref is None:
                if already_reconciled:
                    first_ref = existing
                else:
                    first_ref = ProvisionalQSpec(fields=dict(reconciled_fields))
                    qspecs[slot] = first_ref
                    changed.add(slot)
            else:
                if qspecs.get(slot) is not first_ref:
                    qspecs[slot] = first_ref
                    changed.add(slot)
                if not already_reconciled and first_ref.fields != reconciled_fields:
                    first_ref.fields = dict(reconciled_fields)
                    changed.update(widened)
        if first_ref is not None and first_ref.fields != reconciled_fields:
            first_ref.fields = dict(reconciled_fields)
            changed.update(widened)
        return changed


# ---------------------------------------------------------------------------
# Per-field reconciliation.
# ---------------------------------------------------------------------------


def _reconcile_field(
    field_name: FieldName, slots: frozenset[NodeSlot], qspecs: ProvisionalQSpecMap
) -> FieldValue | None:
    """Compute the reconciled :class:`FieldValue` for one field across ``slots``.

    Returns ``None`` when no slot has a proposal for the field (nothing
    to do). Raises :class:`ReconciliationError` on infeasible fields per
    the per-field policy.
    """
    proposals = [
        qspecs[slot].fields[field_name]
        for slot in slots
        if slot in qspecs and field_name in qspecs[slot].fields
    ]
    if not proposals:
        return None
    policy = _FIELD_POLICY[field_name]
    return policy(proposals)


def _priority_min(proposals: Sequence[FieldValue]) -> int:
    """Highest priority (lowest priority-number) among the proposals."""
    return min(proposal.priority for proposal in proposals)


def _policy_priority_wins(proposals: Sequence[FieldValue]) -> FieldValue:
    """Value from the highest-priority proposal wins. Ties: first encountered."""
    winner = min(proposals, key=lambda proposal: proposal.priority)
    return FieldValue(value=winner.value, priority=_priority_min(proposals))


def _policy_must_agree(proposals: Sequence[FieldValue]) -> FieldValue:
    """All values must be equal, else :class:`ReconciliationError`."""
    values = {proposal.value for proposal in proposals}
    if len(values) > 1:
        raise ReconciliationError(
            f"Incompatible values across slots: {sorted(str(value) for value in values)}. "
            f"Proposals: {[(proposal.value, proposal.priority) for proposal in proposals]}"
        )
    return FieldValue(value=next(iter(values)), priority=_priority_min(proposals))


def _policy_qscheme_lattice(proposals: Sequence[FieldValue]) -> FieldValue:
    """Lattice join: symmetric ⊂ affine; per_tensor ⊂ per_channel. Looser wins."""
    qschemes = [proposal.value for proposal in proposals if proposal.value is not None]
    if not qschemes:
        return FieldValue(value=None, priority=_priority_min(proposals))
    joined = _join_qscheme(set(qschemes))
    return FieldValue(value=joined, priority=_priority_min(proposals))


def _policy_range_min(proposals: Sequence[FieldValue]) -> FieldValue:
    """Union — take the smallest quant_min across proposals."""
    values = [proposal.value for proposal in proposals if proposal.value is not None]
    if not values:
        return FieldValue(value=None, priority=_priority_min(proposals))
    return FieldValue(value=min(values), priority=_priority_min(proposals))


def _policy_range_max(proposals: Sequence[FieldValue]) -> FieldValue:
    """Union — take the largest quant_max across proposals."""
    values = [proposal.value for proposal in proposals if proposal.value is not None]
    if not values:
        return FieldValue(value=None, priority=_priority_min(proposals))
    return FieldValue(value=max(values), priority=_priority_min(proposals))


def _policy_observer_class_lattice(proposals: Sequence[FieldValue]) -> FieldValue:
    """Observer-class lattice: ``fixed ⊂ learning``.

    Fixed observers (``FixedQParamsFakeQuantize`` / ``FixedQParamsObserver``,
    optionally wrapped in a ``.with_args(...)`` partial) are a strict
    specialization of learning observers: they clamp all statistics to a
    baked-in scale / zero_point / range. If a fixed-observer proposal
    ever shares a group with a learning-observer proposal, using the
    fixed one would clamp the group's real observed values to the fixed
    range — usually numerically wrong when the group contains slots
    whose values fall outside the fixed range.

    Rule:

        * If any non-fixed proposal is present, prefer it
          (priority-wins among non-fixed). Fixed proposals that lose
          get a log entry noting the demotion — calibration behavior
          shifts from "clamp to baked range" to "learn observed range".
        * If every proposal is fixed, they must agree on their baked
          parameters (a fixed observer's identity is its partial's
          value); disagreement is a :class:`ReconciliationError`.
    """
    non_fixed = [
        proposal for proposal in proposals if not _is_fixed_observer(proposal.value)
    ]
    fixed = [proposal for proposal in proposals if _is_fixed_observer(proposal.value)]

    if non_fixed:
        winner = min(non_fixed, key=lambda proposal: proposal.priority)
        for demoted in fixed:
            logger.info(
                "OBSERVER_CLASS reconciliation: fixed observer %r (priority %d) "
                "demoted to learning observer %r (priority %d) — group contains "
                "learning-observer members whose values may fall outside the "
                "fixed observer's baked range.",
                demoted.value,
                demoted.priority,
                winner.value,
                winner.priority,
            )
        return FieldValue(value=winner.value, priority=_priority_min(proposals))

    # All fixed — must agree on the exact observer partial (baked
    # scale / zp / range are part of the partial's identity).
    return _policy_must_agree(proposals)


def _is_fixed_observer(observer_class: Any) -> bool:
    """Return True if ``observer_class`` names a fixed-params observer.

    Accepts either a bare class or a ``.with_args(...)`` partial
    (torchao's ``PartialWrapper``). Fixed observers include
    :class:`FixedQParamsFakeQuantize` and :class:`FixedQParamsObserver`
    and any partial whose underlying class is one of those.
    """
    fixed_classes = (FixedQParamsFakeQuantize, FixedQParamsObserver)
    if isinstance(observer_class, type) and issubclass(observer_class, fixed_classes):
        return True
    # Torchao's PartialWrapper exposes the underlying class via ``.p.func``.
    underlying = getattr(getattr(observer_class, "p", None), "func", None)
    if isinstance(underlying, type) and issubclass(underlying, fixed_classes):
        return True
    return False


_FIELD_POLICY: dict[FieldName, Any] = {
    FieldName.DTYPE: _policy_priority_wins,
    FieldName.QSCHEME: _policy_qscheme_lattice,
    FieldName.CH_AXIS: _policy_must_agree,
    FieldName.QUANT_MIN: _policy_range_min,
    FieldName.QUANT_MAX: _policy_range_max,
    FieldName.IS_DYNAMIC: _policy_must_agree,
    FieldName.OBSERVER_CLASS: _policy_observer_class_lattice,
}


def _field_value_stronger(new: FieldValue, current: FieldValue | None) -> bool:
    """Return True iff writing ``new`` would advance the state.

    "Advance" = value differs, or value equals but priority is stronger
    (lower priority-number). Enables no-op detection for convergence.
    """
    if current is None:
        return True
    if new.value != current.value:
        return True
    return new.priority < current.priority


# ---------------------------------------------------------------------------
# qscheme lattice helpers.
# ---------------------------------------------------------------------------


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
PER_CHANNEL_SCHEMES = (torch.per_channel_symmetric, torch.per_channel_affine)


def _join_qscheme(qschemes: set[Any]) -> Any:
    """Lattice join over ``qschemes``: looser wins in symmetry and granularity."""
    is_sym = all(
        _QSCHEME_IS_SYMMETRIC.get(qscheme, False)
        for qscheme in qschemes
        if qscheme is not None
    )
    is_per_channel = any(
        _QSCHEME_IS_PER_CHANNEL.get(qscheme, False) for qscheme in qschemes
    )
    if is_per_channel:
        return torch.per_channel_symmetric if is_sym else torch.per_channel_affine
    return torch.per_tensor_symmetric if is_sym else torch.per_tensor_affine


# ---------------------------------------------------------------------------
# Small helpers callers need.
# ---------------------------------------------------------------------------


def _get_or_create(qspecs: ProvisionalQSpecMap, slot: NodeSlot) -> ProvisionalQSpec:
    """Return the :class:`ProvisionalQSpec` for ``slot``, creating one if absent."""
    if slot not in qspecs:
        qspecs[slot] = ProvisionalQSpec()
    return qspecs[slot]
