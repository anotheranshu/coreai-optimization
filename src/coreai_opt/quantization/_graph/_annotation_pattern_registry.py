# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch.fx.passes.utils.matcher_utils import InternalMatch
from torch.fx.passes.utils.source_matcher_utils import SourcePartition

from coreai_opt._utils.registry_utils import ClassRegistryMixin

from . import _annotation_utils
from ._annotation_utils import (
    OpsListPattern as _OpsListPattern,
)
from ._qspec_constraints import (
    Constraint,
    PER_CHANNEL_SCHEMES,
    ShareFields,
    ShareObserverInstance,
)
from ._qspec_types import (
    FieldName,
    NodeSlot,
    ProvisionalQSpec,
    ProvisionalQSpecMap,
    SlotKind,
)

# Generic type variable for match results
MatchType = TypeVar("MatchType")


@dataclass(frozen=True)
class AnnotatorMatchInfo(Generic[MatchType]):
    """Info about a single annotator's match on the graph.

    Attributes:
        annotator_match (MatchType): Nodes in the model matched with the
            annotator.
        pattern_length (int): Length of the annotation pattern.
    """

    annotator_match: MatchType
    pattern_length: int


def _get_all_patterns_with_activations_appended(starting_pattern: _OpsListPattern):
    """
    Given a starting ops list pattern, return a new list of patterns containing the
    combination of the starting pattern with all activation types appended to the
    pattern.
    """
    patterns = []
    for act_fn in (
        _annotation_utils._supported_activations
        + _annotation_utils._supported_activations_no_inplace
    ):
        patterns.append(_OpsListPattern(starting_pattern.pattern + [act_fn.__name__]))
    return patterns


def _get_all_patterns_from_base_ops(
    base_ops: set[str], use_act: bool = False
) -> list[_OpsListPattern]:
    patterns = []
    for base_op in base_ops:
        if use_act:
            patterns.extend(_get_all_patterns_with_activations_appended(_OpsListPattern([base_op])))
        else:
            patterns.append(_OpsListPattern([base_op]))
    return patterns


class BaseAnnotationPattern(Generic[MatchType], ABC):
    """
    Base class for annotation patterns.

    Each pattern class should implement:
    - generate_patterns(): Returns list of graph modules representing
                           the patterns to match
    - match_single_pattern(): Matches graph for a single pattern
    """

    # Class-level cache for patterns
    _patterns: list[torch.fx.GraphModule | _OpsListPattern] | None = None

    @classmethod
    def get_patterns(cls) -> list[torch.fx.GraphModule | _OpsListPattern]:
        """
        Returns a cached list of graph modules representing the patterns
        this annotator can match. Patterns are generated once and cached.
        All patterns associated with a BaseAnnotationPattern subclass must
        contain the same number of operations. For example, a subclass can
        contain patterns for Conv1d/2d/3d -> Relu (2 operations), but cannot
        contain a Conv2d -> Relu pattern as well as a Conv2d only pattern.
        Define another BaseAnnotationPattern subclass in such cases.

        Returns:
            List of torch.fx.GraphModule objects representing different patterns
        """
        if cls._patterns is None:
            cls._patterns = cls.generate_patterns()
            cls._validate_patterns_length()
        return cls._patterns

    @classmethod
    def _validate_patterns_length(cls):
        """Ensure that all patterns for the class have the same length"""
        if not cls._patterns:
            raise RuntimeError(f"{cls.__name__} has no associated patterns.")
        first_pattern_length = cls._get_single_pattern_length(cls._patterns[0])
        for pattern in cls._patterns[1:]:
            if cls._get_single_pattern_length(pattern) != first_pattern_length:
                raise RuntimeError(
                    f"Expected all patterns to have the same length for class {cls.__name__}"
                )

    @staticmethod
    def _get_single_pattern_length(pattern):
        pattern_len = 0
        # Account for AnnotationPattern classes using get_sequential_partition with a
        # list of types.
        if isinstance(pattern, _OpsListPattern):
            return len(pattern.pattern)
        for node in pattern.graph.nodes:
            # Sanity check to make sure we account for all node.op types
            assert node.op in [
                "get_attr",
                "placeholder",
                "output",
                "call_function",
                "call_method",
                "call_module",
            ]
            if node.op not in ["get_attr", "placeholder", "output"]:
                pattern_len += 1
        return pattern_len

    @classmethod
    def get_pattern_length(cls):
        if not cls.get_patterns():
            return 0
        return cls._get_single_pattern_length(cls.get_patterns()[0])

    @classmethod
    @abstractmethod
    def generate_patterns(cls) -> list[torch.fx.GraphModule | _OpsListPattern]:
        """
        Generate the patterns for this annotator. Called once and cached.

        Returns:
            List of torch.fx.GraphModule or _OpsListPattern objects representing
            different patterns
        """
        pass

    @classmethod
    @abstractmethod
    def match_single_pattern(
        cls, model: torch.fx.GraphModule, pattern: torch.fx.GraphModule | _OpsListPattern
    ) -> dict[torch.fx.Node, MatchType]:
        """
        Match nodes in model to the provided pattern and return a dictionary mapping
        nodes to matches.

        Args:
            model (torch.fx.GraphModule): Exported GraphModule model to match
            pattern (torch.fx.GraphModule | _OpsListPattern): Pattern used to match

        Returns:
            dict[torch.fx.Node, MatchType]: Dictionary mapping nodes to matches.
        """
        raise NotImplementedError

    @classmethod
    def _match_all_patterns(
        cls, model: torch.fx.GraphModule
    ) -> dict[torch.fx.Node, AnnotatorMatchInfo]:
        """
        Match all patterns associated with the annotation to the model and return a
        dictionary mapping nodes in the model to AnnotatorMatchInfo containing info
        on the pattern and matches.
        """
        all_nodes_to_match_dict: dict[torch.fx.Node, AnnotatorMatchInfo] = {}
        patterns = cls.get_patterns()

        for pattern in patterns:
            # Pass a copy of cached pattern as downstream operations
            # modify the graph module inplace
            pattern_copy = deepcopy(pattern)
            node_to_match_dict = cls.match_single_pattern(model, pattern_copy)
            # Note: It is possible for a node in node_to_match_dict to already exist
            # in all_nodes_to_match_dict due to multiple patterns being able to
            # match the same node. For example, 'add_' and 'add' both match add node.
            # We do a sanity check to ensure that the expectation holds that the matched
            # nodes align in these cases.
            intersecting_nodes = set(all_nodes_to_match_dict) & set(node_to_match_dict)
            for node in intersecting_nodes:
                # Sanity check that intersections should always match.
                error_msg = (
                    "Duplicate node with differing match dict found. Node: {node}, "
                    f"first match_dict: "
                    f"{all_nodes_to_match_dict[node].annotator_match}, "
                    f"second match_dict: {node_to_match_dict[node]}."
                )
                assert all_nodes_to_match_dict[node].annotator_match == node_to_match_dict[node], (
                    error_msg
                )

            # For each node corresponding to a match, create AnnotatorMatchInfo to
            # track additional annotator info along with the match itself
            all_nodes_to_match_dict.update(
                {
                    node: AnnotatorMatchInfo(
                        pattern_length=cls.get_pattern_length(),
                        annotator_match=match,
                    )
                    for node, match in node_to_match_dict.items()
                }
            )
        return all_nodes_to_match_dict


class WeightedModulePattern(BaseAnnotationPattern[InternalMatch]):
    """
    Base class for annotation patterns that handle modules with weights.

    This class provides a common implementation for patterns involving modules
    that have weights, such as convolution, linear, and embedding layers.
    Bias is optional and may or may not be present.

    It handles patterns of the form:

        input -> weighted_module -> [batch_norm] -> [activation] -> output

    Where batch_norm and activation are optional components that may or may not
    be present in the pattern.
    """

    @classmethod
    def match_single_pattern(
        cls, model: torch.fx.GraphModule, pattern: torch.fx.GraphModule | _OpsListPattern
    ) -> dict[torch.fx.Node, InternalMatch]:
        """
        Match nodes in model to the provided pattern and return a dictionary mapping
        nodes to matches.
        """
        if not isinstance(pattern, torch.fx.GraphModule):
            error_msg = (
                "WeightedModulePattern currently supports patterns of torch.fx.GraphModule only"
            )
            raise NotImplementedError(error_msg)

        return _annotation_utils.match_pattern_with_subgraph_matcher(model, pattern)


class NAryActPattern(BaseAnnotationPattern[tuple[SourcePartition]]):
    """
    Base class for annotation patterns that handle an op of one or more inputs along
    with an optional activation.

    It handles patterns of the form:

        input1/[input2, ...] -> op1 -> [act]

    The annotation logic expects to match 1 or 2 ops. If there are 2 ops, the second op
    is expected to be an activation.
    The initial op can have one or more inputs, where all inputs are expected to be
    quantized.
    No handling for weights/parameters/attributes is done.
    No quantization will take place before the activation if one exists.

    Subclasses must implement the generate_patterns() method to define the
    specific patterns they want to match (e.g., MatMul, MatMul+activation,
    Add, Add+activation, etc.).
    """

    @classmethod
    def match_single_pattern(
        cls, model: torch.fx.GraphModule, pattern: torch.fx.GraphModule | _OpsListPattern
    ) -> dict[torch.fx.Node, tuple[SourcePartition]]:
        """
        Match nodes in model to the provided pattern and return a dictionary mapping
        nodes to matches.
        """
        if not isinstance(pattern, _OpsListPattern):
            error_msg = (
                "NAryActPattern expects patterns of type _OpsListPattern, but "
                f"got type {type(pattern)}.)"
            )
            raise RuntimeError(error_msg)

        return _annotation_utils.match_pattern_with_sequential_partitions(model, pattern)


class SharedObserverModulePattern(BaseAnnotationPattern[tuple[SourcePartition]]):
    """
    Base class for annotation patterns that handle shared observer modules.

    This class provides a common implementation for patterns involving modules
    which require shared observers for input and output.

    It handles patterns of the form:

        input -> shared observer op -> output

    Subclasses must implement the generate_patterns() method to define the
    specific patterns they want to match (e.g., maxpool, avgpool, flatten, etc.).

    Subclasses must also implement :meth:`generate_qspec_sharing_constraints`
    to declare how quantization specs are shared across the op's input and
    output slots. Missing this override raises ``TypeError`` at instantiation.
    """

    @classmethod
    @abstractmethod
    def generate_qspec_sharing_constraints(
        cls,
        node: torch.fx.Node,
        qspecs: ProvisionalQSpecMap,
    ) -> list[Constraint]:
        """Return the constraints that express this op's qspec-sharing semantics.

        Called by the reconciler once per matched shared-observer node with
        the current provisional-qspec state. Each subclass owns the decision
        of which of its input / output slots must share dtype, is_dynamic,
        or a full observer instance.

        Args:
            node (torch.fx.Node): The matched op node whose slots the
                returned constraints reference.
            qspecs (ProvisionalQSpecMap): Current provisional-qspec state,
                available for dynamic peeking (e.g. concat's axis-aware
                observer-sharing decision reads the current output-slot
                qscheme and ch_axis).

        Returns:
            list[Constraint]: Constraints to enqueue. Empty list means
            "no sharing this op wants to enforce."
        """
        raise NotImplementedError

    @classmethod
    def _validate_patterns_length(cls):
        super()._validate_patterns_length()
        pattern_len = cls._get_single_pattern_length(cls._patterns[0])
        if pattern_len != 1:
            error_msg = (
                "Shared observer patterns must have a pattern of length 1, "
                f"however got length {pattern_len} for pattern {cls.__name__}."
            )
            raise RuntimeError(error_msg)

    @classmethod
    def match_single_pattern(
        cls, model: torch.fx.GraphModule, pattern: torch.fx.GraphModule | _OpsListPattern
    ) -> dict[torch.fx.Node, tuple[SourcePartition]]:
        """
        Match nodes in model to the provided pattern and return a dictionary mapping
        nodes to matches.
        """
        if not isinstance(pattern, _OpsListPattern):
            error_msg = (
                "SharedObserverModulePattern expects patterns of type "
                f"_OpsListPattern, but got type {type(pattern)}.)"
            )
            raise RuntimeError(error_msg)

        return _annotation_utils.match_pattern_with_sequential_partitions(model, pattern)


def _shared_op_boundary_slots(node: torch.fx.Node) -> tuple[NodeSlot, list[NodeSlot]]:
    """Return the output slot and list of input slots on a shared-observer node.

    Utility for :class:`SharedObserverModulePattern` subclass implementations
    of :meth:`generate_qspec_sharing_constraints`.
    """
    output_slot = NodeSlot(node=node, kind=SlotKind.OUTPUT, arg_index=0)
    input_slots = [
        NodeSlot(node=node, kind=SlotKind.INPUT, arg_index=arg_index)
        for arg_index in range(len(node.all_input_nodes))
    ]
    return output_slot, input_slots


def _any_input_slot_populated(
    input_slots: list[NodeSlot], qspecs: ProvisionalQSpecMap
) -> bool:
    """Return True iff any of the input slots already has a proposal in ``qspecs``.

    The shared-observer sharing rule for cat / maxpool / etc. only fires when
    at least one input side wants observation — otherwise there's nothing
    for the sharing to enforce.
    """
    return any(input_slot in qspecs for input_slot in input_slots)


class _AnnotationPatternRegistry(ClassRegistryMixin):
    """
    A registry of quantization annotation pattern classes.
    """


@_AnnotationPatternRegistry.register("conv_bn_act")
class ConvBNActPattern(WeightedModulePattern):
    """
    Annotates input -> conv -> bn -> activation -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[torch.fx.GraphModule]:
        """Returns all conv-activation pattern variations."""
        patterns = []
        for conv_dim in [1, 2, 3]:
            for act_fn in _annotation_utils._supported_activations:
                for act_in_place in [True, False]:
                    pattern_gm = _annotation_utils.get_conv_bn_pattern(
                        conv_dim=conv_dim,
                        is_transpose=False,
                        act_fn=act_fn,
                        act_in_place=act_in_place,
                    )
                    patterns.append(pattern_gm)

            # Add patterns for activations without inplace support
            for act_fn in _annotation_utils._supported_activations_no_inplace:
                pattern_gm = _annotation_utils.get_conv_bn_pattern(
                    conv_dim=conv_dim, is_transpose=False, act_fn=act_fn, act_in_place=False
                )
                patterns.append(pattern_gm)
        return patterns


@_AnnotationPatternRegistry.register("conv_transpose_bn_act")
class ConvTransposeBNActPattern(WeightedModulePattern):
    """
    Annotates input -> conv_transpose -> bn -> activation -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[torch.fx.GraphModule]:
        """Returns all conv_transpose-activation pattern variations."""
        patterns = []
        for conv_dim in [1, 2, 3]:
            for act_fn in _annotation_utils._supported_activations:
                for act_in_place in [True, False]:
                    pattern_gm = _annotation_utils.get_conv_bn_pattern(
                        conv_dim=conv_dim,
                        is_transpose=True,
                        act_fn=act_fn,
                        act_in_place=act_in_place,
                    )
                    patterns.append(pattern_gm)

            # Add patterns for activations without inplace support
            for act_fn in _annotation_utils._supported_activations_no_inplace:
                pattern_gm = _annotation_utils.get_conv_bn_pattern(
                    conv_dim=conv_dim, is_transpose=True, act_fn=act_fn, act_in_place=False
                )
                patterns.append(pattern_gm)

        return patterns


@_AnnotationPatternRegistry.register("conv_act")
class ConvActPattern(WeightedModulePattern):
    """
    Annotates input -> conv -> activation -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[torch.fx.GraphModule]:
        """Returns all conv-activation pattern variations."""
        patterns = []
        for conv_dim in [1, 2, 3]:
            for act_fn in _annotation_utils._supported_activations:
                for act_in_place in [True, False]:
                    pattern_gm = _annotation_utils.get_conv_pattern(
                        conv_dim=conv_dim, act_fn=act_fn, act_in_place=act_in_place
                    )
                    patterns.append(pattern_gm)

            # Add patterns for activations without inplace support
            for act_fn in _annotation_utils._supported_activations_no_inplace:
                pattern_gm = _annotation_utils.get_conv_pattern(
                    conv_dim=conv_dim, act_fn=act_fn, act_in_place=False
                )
                patterns.append(pattern_gm)

        return patterns


@_AnnotationPatternRegistry.register("conv_transpose_act")
class ConvTransposeActPattern(WeightedModulePattern):
    """
    Annotates input -> conv_transpose -> activation -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[torch.fx.GraphModule]:
        """Returns all conv_transpose-activation pattern variations."""
        patterns = []
        for conv_dim in [1, 2, 3]:
            for act_fn in _annotation_utils._supported_activations:
                for act_in_place in [True, False]:
                    pattern_gm = _annotation_utils.get_conv_pattern(
                        conv_dim=conv_dim,
                        is_transpose=True,
                        act_fn=act_fn,
                        act_in_place=act_in_place,
                    )
                    patterns.append(pattern_gm)

            # Add patterns for activations without inplace support
            for act_fn in _annotation_utils._supported_activations_no_inplace:
                pattern_gm = _annotation_utils.get_conv_pattern(
                    conv_dim=conv_dim, is_transpose=True, act_fn=act_fn, act_in_place=False
                )
                patterns.append(pattern_gm)

        return patterns


@_AnnotationPatternRegistry.register("conv_bn")
class ConvBNPattern(WeightedModulePattern):
    """
    Annotates input -> conv -> bn -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[torch.fx.GraphModule]:
        """Returns all conv pattern variations."""
        patterns = []
        for conv_dim in [1, 2, 3]:
            pattern_gm = _annotation_utils.get_conv_bn_pattern(
                conv_dim=conv_dim, is_transpose=False, act_fn=None
            )
            patterns.append(pattern_gm)
        return patterns


@_AnnotationPatternRegistry.register("conv_transpose_bn")
class ConvTransposeBNPattern(WeightedModulePattern):
    """
    Annotates input -> conv_transpose -> bn -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[torch.fx.GraphModule]:
        """Returns all conv_transpose pattern variations."""
        patterns = []
        for conv_dim in [1, 2, 3]:
            pattern_gm = _annotation_utils.get_conv_bn_pattern(
                conv_dim=conv_dim, is_transpose=True, act_fn=None
            )
            patterns.append(pattern_gm)
        return patterns


@_AnnotationPatternRegistry.register("conv")
class ConvPattern(WeightedModulePattern):
    """
    Annotates input -> conv -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[torch.fx.GraphModule]:
        """Returns all conv pattern variations."""
        patterns = []
        for conv_dim in [1, 2, 3]:
            pattern_gm = _annotation_utils.get_conv_pattern(conv_dim=conv_dim, act_fn=None)
            patterns.append(pattern_gm)
        return patterns


@_AnnotationPatternRegistry.register("conv_transpose")
class ConvTransposePattern(WeightedModulePattern):
    """
    Annotates input -> conv_transpose -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[torch.fx.GraphModule]:
        """Returns all conv_transpose pattern variations."""
        patterns = []
        for conv_dim in [1, 2, 3]:
            pattern_gm = _annotation_utils.get_conv_pattern(
                conv_dim=conv_dim, is_transpose=True, act_fn=None
            )
            patterns.append(pattern_gm)
        return patterns


@_AnnotationPatternRegistry.register("linear_bn_act")
class LinearBNActPattern(WeightedModulePattern):
    """
    Annotates input -> linear -> bn -> activation -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[torch.fx.GraphModule]:
        """Returns all linear-activation pattern variations."""
        patterns = []
        for act_fn in _annotation_utils._supported_activations:
            for act_in_place in [True, False]:
                pattern_gm = _annotation_utils.get_linear_bn_pattern(
                    act_fn=act_fn, act_in_place=act_in_place
                )
                patterns.append(pattern_gm)

        # Add patterns for activations without inplace support
        for act_fn in _annotation_utils._supported_activations_no_inplace:
            pattern_gm = _annotation_utils.get_linear_bn_pattern(act_fn=act_fn, act_in_place=False)
            patterns.append(pattern_gm)

        return patterns


@_AnnotationPatternRegistry.register("linear_act")
class LinearActPattern(WeightedModulePattern):
    """
    Annotates input -> linear -> activation -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[torch.fx.GraphModule]:
        """Returns all linear-activation pattern variations."""
        patterns = []
        for act_fn in _annotation_utils._supported_activations:
            for act_in_place in [True, False]:
                pattern_gm = _annotation_utils.get_linear_pattern(
                    act_fn=act_fn, act_in_place=act_in_place
                )
                patterns.append(pattern_gm)

        # Add patterns for activations without inplace support
        for act_fn in _annotation_utils._supported_activations_no_inplace:
            pattern_gm = _annotation_utils.get_linear_pattern(act_fn=act_fn, act_in_place=False)
            patterns.append(pattern_gm)

        return patterns


@_AnnotationPatternRegistry.register("linear_bn")
class LinearBNPattern(WeightedModulePattern):
    """
    Annotates input -> linear -> bn -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[torch.fx.GraphModule]:
        """Returns linear pattern."""
        pattern_gm = _annotation_utils.get_linear_bn_pattern(act_fn=None)
        return [pattern_gm]


@_AnnotationPatternRegistry.register("linear")
class LinearPattern(WeightedModulePattern):
    """
    Annotates input -> linear -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[torch.fx.GraphModule]:
        """Returns linear pattern."""
        pattern_gm = _annotation_utils.get_linear_pattern(act_fn=None)
        return [pattern_gm]


@_AnnotationPatternRegistry.register("embedding")
class EmbeddingPattern(WeightedModulePattern):
    """
    Annotates input -> embedding -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[torch.fx.GraphModule]:
        """Returns embedding pattern."""
        pattern_gm = _annotation_utils.get_embedding_pattern()
        return [pattern_gm]


@_AnnotationPatternRegistry.register("matmul")
class MatMulPattern(NAryActPattern):
    """
    Annotates input -> matmul -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[_OpsListPattern]:
        """Returns matmul pattern."""
        return _get_all_patterns_from_base_ops({"matmul"})


@_AnnotationPatternRegistry.register("matmul_act")
class MatMulActPattern(NAryActPattern):
    """
    Annotates input -> matmul -> act -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[_OpsListPattern]:
        """Returns matmul act pattern."""
        return _get_all_patterns_from_base_ops({"matmul"}, use_act=True)


@_AnnotationPatternRegistry.register("add")
class AddPattern(NAryActPattern):
    """
    Annotates input -> add -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[_OpsListPattern]:
        """Returns add pattern."""
        return _get_all_patterns_from_base_ops({"add", "add_"})


@_AnnotationPatternRegistry.register("add_act")
class AddActPattern(NAryActPattern):
    """
    Annotates input -> add -> act -> output
    """

    @classmethod
    def generate_patterns(cls):
        return _get_all_patterns_from_base_ops({"add", "add_"}, use_act=True)


@_AnnotationPatternRegistry.register("mul")
class MulPattern(NAryActPattern):
    """
    Annotates input -> mul -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[_OpsListPattern]:
        """Returns mul pattern."""
        return _get_all_patterns_from_base_ops({"mul", "mul_"})


@_AnnotationPatternRegistry.register("mul_act")
class MulActPattern(NAryActPattern):
    """
    Annotates input -> mul -> act -> output
    """

    @classmethod
    def generate_patterns(cls):
        return _get_all_patterns_from_base_ops({"mul", "mul_"}, use_act=True)


@_AnnotationPatternRegistry.register("sub")
class SubPattern(NAryActPattern):
    """
    Annotates input -> sub -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[_OpsListPattern]:
        """Returns sub pattern."""
        # Multiple types of sub need to be listed since torchao doesn't include sub in
        # pt2e.graph_utils._EQUIVALENT_TYPES dict
        return _get_all_patterns_from_base_ops({"sub", "sub_", "isub"})


def _make_kv_cache_update_pattern(op_name: str) -> type[NAryActPattern]:
    """Build a pattern class that annotates the named KV-cache update op.

    Returns a fresh class on every call so per-run state stays scoped to a
    single ``Quantizer`` instance instead of leaking into the process-global
    ``_AnnotationPatternRegistry``.
    """

    class KVCacheUpdatePattern(NAryActPattern):
        """input -> kv-cache-update op -> output"""

        @classmethod
        def generate_patterns(cls) -> list[_OpsListPattern]:
            return _get_all_patterns_from_base_ops({op_name}, use_act=False)

    return KVCacheUpdatePattern


# Shared observer patterns
@_AnnotationPatternRegistry.register("flatten")
class FlattenPattern(SharedObserverModulePattern):
    """
    Annotates input -> flatten -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[torch.fx.GraphModule]:
        """Returns flatten pattern."""
        return _get_all_patterns_from_base_ops({"flatten"})

    @classmethod
    def generate_qspec_sharing_constraints(
        cls, node: torch.fx.Node, qspecs: ProvisionalQSpecMap
    ) -> list[Constraint]:
        """All slots share one observer instance — flatten preserves values."""
        return _all_slots_share_observer(node, qspecs)


@_AnnotationPatternRegistry.register("maxpool")
class MaxPoolPattern(SharedObserverModulePattern):
    """
    Annotates input -> maxpool1d/2d/3d -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[torch.fx.GraphModule]:
        """Returns maxpool pattern."""
        return _get_all_patterns_from_base_ops(
            {
                "max_pool1d",
                "max_pool2d",
                "max_pool3d",
            }
        )

    @classmethod
    def generate_qspec_sharing_constraints(
        cls, node: torch.fx.Node, qspecs: ProvisionalQSpecMap
    ) -> list[Constraint]:
        """All slots share one observer instance — maxpool preserves values."""
        return _all_slots_share_observer(node, qspecs)


@_AnnotationPatternRegistry.register("avgpool")
class AvgPoolPattern(SharedObserverModulePattern):
    """
    Annotates input -> avgpool1d/2d/3d -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[torch.fx.GraphModule]:
        """Returns avgpool pattern."""
        return _get_all_patterns_from_base_ops(
            {
                "avg_pool1d",
                "avg_pool2d",
                "avg_pool3d",
                "adaptive_avg_pool1d",
                "adaptive_avg_pool2d",
                "adaptive_avg_pool3d",
                "mean",
            }
        )

    @classmethod
    def generate_qspec_sharing_constraints(
        cls, node: torch.fx.Node, qspecs: ProvisionalQSpecMap
    ) -> list[Constraint]:
        """All slots share one observer instance — averaging preserves range roughly."""
        return _all_slots_share_observer(node, qspecs)


@_AnnotationPatternRegistry.register("concat")
class ConcatPattern(SharedObserverModulePattern):
    """
    Annotates input/input2/... -> cat/concat -> output
    """

    @classmethod
    def generate_patterns(cls) -> list[torch.fx.GraphModule]:
        """Returns concat pattern."""
        return _get_all_patterns_from_base_ops({"cat", "concat"})

    @classmethod
    def generate_qspec_sharing_constraints(
        cls, node: torch.fx.Node, qspecs: ProvisionalQSpecMap
    ) -> list[Constraint]:
        """Axis-aware sharing:

        * All slots always share ``DTYPE`` and ``IS_DYNAMIC``.
        * If the reconciled output qscheme is per-channel along the concat
          dimension, each input keeps its own scale — no observer-instance
          sharing needed. Per-channel along a *different* axis (or per-tensor)
          requires all inputs to observe together.
        """
        output_slot, input_slots = _shared_op_boundary_slots(node)
        if not _any_input_slot_populated(input_slots, qspecs):
            return []

        all_slots = frozenset([output_slot, *input_slots])
        constraints: list[Constraint] = [
            ShareFields(
                _slots=all_slots,
                fields=frozenset({FieldName.DTYPE, FieldName.IS_DYNAMIC}),
            )
        ]

        concat_dim = _concat_dim(node)
        output_qspec = qspecs.get(output_slot, ProvisionalQSpec()).fields
        output_qscheme = output_qspec.get(FieldName.QSCHEME)
        output_ch_axis = output_qspec.get(FieldName.CH_AXIS)

        per_channel_along_concat_axis = (
            output_qscheme is not None
            and output_qscheme.value in PER_CHANNEL_SCHEMES
            and output_ch_axis is not None
            and output_ch_axis.value == concat_dim
        )
        if not per_channel_along_concat_axis:
            constraints.append(ShareObserverInstance(_slots=all_slots))
        return constraints


def _all_slots_share_observer(
    node: torch.fx.Node, qspecs: ProvisionalQSpecMap
) -> list[Constraint]:
    """Emit ``ShareObserverInstance`` across the op's input + output slots.

    Used by shared-observer ops (flatten, maxpool, avgpool) whose sharing
    semantics don't depend on any spec field — they always tie every input
    and output slot into one observer instance when at least one input
    side wants observation.
    """
    output_slot, input_slots = _shared_op_boundary_slots(node)
    if not _any_input_slot_populated(input_slots, qspecs):
        return []
    return [ShareObserverInstance(_slots=frozenset([output_slot, *input_slots]))]


def _concat_dim(node: torch.fx.Node) -> int:
    """Concat dim from ``aten.cat(tensors, dim=?)`` (defaults to 0)."""
    if len(node.args) >= 2:
        return int(node.args[1])
    return int(node.kwargs.get("dim", 0))
