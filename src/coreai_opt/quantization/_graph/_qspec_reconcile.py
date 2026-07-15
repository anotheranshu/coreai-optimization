# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Constraint-queue reconciliation for graph-mode quantization annotation.

Driver for the full pipeline. Types, per-field policies, constraint
generation, and resolution live in sibling modules:

    * ``_qspec_types`` — :class:`NodeSlot`, :class:`FieldName`,
      :class:`ProvisionalQSpec`, :class:`ProvisionalQSpecMap`, etc.
    * ``_qspec_constraints`` — :class:`Constraint` ABC, per-field
      policies, :func:`_reconcile_field`.
    * ``_provisional_qspec_generation`` — :func:`build_initial_state`.
    * ``_qspec_constraint_generation`` — :class:`_AnnotationContext`,
      :func:`_generate_constraints_for_node`,
      :func:`_nodes_covered_by`.
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

from collections import deque

import torch.fx as fx

from ._provisional_qspec_generation import build_initial_provisional_qspecs
from ._qspec_constraint_generation import (
    _AnnotationContext,
    _generate_constraints_for_node,
    _nodes_covered_by,
)
from ._qspec_constraints import Constraint
from ._qspec_resolution import resolve_qspecs

# Re-export so callers depend only on _qspec_reconcile.
__all__ = [
    "_AnnotationContext",
    "_nodes_covered_by",
    "annotate_via_reconciliation",
]


def annotate_via_reconciliation(
    model: fx.GraphModule, ctx: _AnnotationContext
) -> fx.GraphModule:
    """Annotate ``model`` in place using the constraint-queue reconciler."""
    qspecs = build_initial_provisional_qspecs(
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
