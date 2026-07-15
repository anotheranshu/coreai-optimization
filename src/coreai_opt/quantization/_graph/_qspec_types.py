# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Shared types for the qspec reconciliation pipeline.

These types are the vocabulary every phase of the pipeline speaks —
provisional-qspec generation, constraint generation + drain, and
resolution onto per-node annotations. Keeping them in a small,
dependency-free module lets each phase live in its own file without
importing sideways into another phase.

Nothing here is behavior; only data + enums + one exception type.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import torch.fx as fx


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

    Each field has its own reconciliation policy — see
    ``_qspec_constraints._FIELD_POLICY``.
    """

    DTYPE = enum.auto()
    QSCHEME = enum.auto()  # combined symmetry × granularity — one lattice-joined field
    CH_AXIS = enum.auto()
    QUANT_MIN = enum.auto()
    QUANT_MAX = enum.auto()
    IS_DYNAMIC = enum.auto()
    OBSERVER_CLASS = enum.auto()


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
    ``ShareObserverInstance`` expresses sharing: two slots point at the
    same Python object, so any field mutation is seen by both. Object
    identity IS the sharing relation; there's no separate side-list.
    """

    fields: dict[FieldName, FieldValue] = field(default_factory=dict)


ProvisionalQSpecMap = dict[NodeSlot, ProvisionalQSpec]
"""Alias for the map every phase reads/writes.

Two slots pointing at the same :class:`ProvisionalQSpec` object share an
observer at runtime. Two slots pointing at distinct objects reconcile
independently.
"""


# ---------------------------------------------------------------------------
# Cross-phase exception.
# ---------------------------------------------------------------------------


class ReconciliationError(RuntimeError):
    """Raised when reconciliation hits an unresolvable state.

    Fatal cases:

        * ``IS_DYNAMIC`` disagreement across slots.
        * Incompatible ``CH_AXIS`` values across per_channel slots.
        * All-fixed ``OBSERVER_CLASS`` group whose fixed params disagree.
        * Pattern-structure violations detected during provisional-qspec
          generation (see ``_provisional_qspec_generation.py``).
    """
