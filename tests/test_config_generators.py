import pytest
from dataclasses import dataclass, field
from typing import Optional
from degann.networks.topology.tuning_utils import (
    TuningMetadata,
    FieldMetadata,
    generate_all_configurations,
)


@pytest.mark.parametrize(
    "value_range, expected_values",
    [
        # Standard integer range with step size 1
        ((1, 3, 1), {1, 2, 3}),
        # Single-value edge case (min == max)
        ((5, 5, 1), {5}),
    ],
)
def test_int_range_config(value_range, expected_values):
    """
    Test integer value range generation with various boundary conditions.
    Validates:
    - Standard ranges with multiple values
    - Single-value ranges
    - Step size handling for integers
    """

    @dataclass
    class IntConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        a: int = field(default=2, metadata={"tunable": True})

    tm = TuningMetadata(IntConfig)
    tm.set_metadata({"a": FieldMetadata(value_range=value_range)})

    int_config = IntConfig(tuning_metadata=tm)
    candidates = list(generate_all_configurations(int_config))

    result_values = {c.a for c in candidates}

    assert result_values == expected_values


@pytest.mark.parametrize(
    "value_range, expected_values",
    [
        # Floating point range with fractional step
        ((0.0, 0.2, 0.1), {0.0, 0.1, 0.2}),
        # Single-value edge case with zero range
        ((0.0, 0.0, 0.5), {0.0}),
    ],
)
def test_float_range_config(value_range, expected_values):
    """
    Test floating point value range generation.
    Validates:
    - Handling of floating point increments
    - Proper inclusion of endpoint values
    """

    @dataclass
    class FloatConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        a: float = field(default=2.0, metadata={"tunable": True})

    tm = TuningMetadata(FloatConfig)
    tm.set_metadata({"a": FieldMetadata(value_range=value_range)})

    float_config = FloatConfig(tuning_metadata=tm)
    candidates = list(generate_all_configurations(float_config))

    result_values = {c.a for c in candidates}

    assert result_values == expected_values


@pytest.mark.parametrize(
    "choices, expected_values",
    [
        # Normal case with multiple choices
        (("A", "B", "C"), {"A", "B", "C"}),
        # No choices provided (fallback to default)
        (None, {"default"}),
        # Empty choices list (should use default)
        ([], {"default"}),
        # Single choice edge case
        (["X"], {"X"}),
    ],
)
def test_choice_config(choices, expected_values):
    """
    Test choice-based configuration generation.
    Validates:
    - Multiple choice selection
    - Empty/NULL choice handling
    - Default value fallback behavior
    - Single-choice edge cases
    """

    @dataclass
    class ChoiceConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        c: str = field(default="default", metadata={"tunable": True})

    tm = TuningMetadata(ChoiceConfig)
    tm.set_metadata({"c": FieldMetadata(choices=choices)})

    config = ChoiceConfig(
        tuning_metadata=tm,
    )
    candidates = list(generate_all_configurations(config))

    result_values = {instance.c for instance in candidates}

    assert result_values == expected_values


@pytest.mark.parametrize(
    "value_range, choices, expected_values",
    [
        # Combined value range and choices
        (
            (0.0, 3.0, 1.0),  # Generates [0.0, 1.0, 2.0, 3.0]
            (0.5, 1.5, 2.5),  # Direct choices
            {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0},  # Combined set
        )
    ],
)
def test_mixed_config(value_range, choices, expected_values):
    """
    Test configuration generation with both value ranges and choices.
    Validates:
    - Combined value sources (range + choices)
    """

    @dataclass
    class MixedConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        c: float = field(default=5.0, metadata={"tunable": True})

    tm = TuningMetadata(MixedConfig)
    tm.set_metadata({"c": FieldMetadata(choices=choices, value_range=value_range)})

    config = MixedConfig(
        tuning_metadata=tm,
    )
    candidates = list(generate_all_configurations(config))

    result_values = {instance.c for instance in candidates}

    assert result_values == expected_values


def test_no_tuning_metadata():
    """
    Test behavior when no tuning metadata exists.
    Validates:
    - Fallback to default configuration
    - No candidate generation when metadata is missing
    """

    @dataclass
    class NoTuneConfig:
        a: int = field(default=10, metadata={"tunable": True})
        b: str = field(default="test", metadata={"tunable": True})

    config = NoTuneConfig()
    candidates = list(generate_all_configurations(config))

    # Should return only one candidate: the original configuration.
    assert len(candidates) == 1
    assert candidates[0].a == 10
    assert candidates[0].b == "test"


def test_empty_metadata():
    """
    Test behavior with initialized but empty metadata.
    Validates:
    - Default FieldMetadata handling
    - Empty metadata initialization
    - No candidate expansion when no constraints exist
    """

    @dataclass
    class EmptyMetadataConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        a: int = field(default=10, metadata={"tunable": True})
        b: str = field(default="test", metadata={"tunable": True})

    tm = TuningMetadata(EmptyMetadataConfig)

    config = EmptyMetadataConfig(tuning_metadata=tm)
    candidates = list(generate_all_configurations(config))

    # Should return only one candidate: the original configuration.
    assert len(candidates) == 1
    assert candidates[0].a == 10
    assert candidates[0].b == "test"


@pytest.mark.parametrize(
    "metadata_dict, configurations",
    [
        (
            {
                "a": FieldMetadata(value_range=(1, 2, 1)),  # [1,2]
                "b": FieldMetadata(choices=[10, 20]),  # [[10], [20]]
                "c": FieldMetadata(choices=["x", "y"]),  # ["x", "y"]
            },
            # Cartesian product of all combinations
            {
                (1, (10,), "x"),
                (1, (10,), "y"),
                (1, (20,), "x"),
                (1, (20,), "y"),
                (2, (10,), "x"),
                (2, (10,), "y"),
                (2, (20,), "x"),
                (2, (20,), "y"),
            },
        )
    ],
)
def test_mixed_fields(metadata_dict, configurations):
    """
    Test configuration generation with multiple field types.
    Validates:
    - Interaction between different field types (int, list, str)
    - Cartesian product generation across multiple fields
    - Proper nesting of list-type fields
    """

    @dataclass
    class MixedConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        a: int = field(default=0, metadata={"tunable": True})
        b: list[int] = field(default_factory=lambda: [10], metadata={"tunable": True})
        c: str = field(default="x", metadata={"tunable": True})

    tm = TuningMetadata(MixedConfig)
    tm.set_metadata(metadata_dict)

    config = MixedConfig(tuning_metadata=tm)
    candidates = list(generate_all_configurations(config))

    generated_configs = {(cfg.a, tuple(cfg.b), cfg.c) for cfg in candidates}

    assert generated_configs == configurations


@pytest.mark.parametrize(
    "value_range, length_boundary, expected_candidates",
    [
        # Fixed length list generation from value range
        ((10, 20, 10), (1, 1), {(10,), (20,)}),
        # Variable length list generation with combinations
        ((10, 20, 10), (1, 2), {(10, 10), (10, 20), (20, 10), (20, 20), (10,), (20,)}),
    ],
)
def test_list_with_value_range_only(value_range, length_boundary, expected_candidates):
    """
    Test list generation using value ranges with length constraints.
    Validates:
    - Fixed-length list generation
    - Variable-length list combinations
    - Proper application of length boundaries
    - Cartesian product generation for list elements
    """

    @dataclass
    class ListRangeConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        b: list[int] = field(default_factory=lambda: [10], metadata={"tunable": True})

    tm = TuningMetadata(ListRangeConfig)
    tm.set_metadata(
        {"b": FieldMetadata(value_range=value_range, length_boundary=length_boundary)}
    )

    config = ListRangeConfig(tuning_metadata=tm)
    candidates = list(generate_all_configurations(config))

    result = {tuple(c.b) for c in candidates}
    assert result == expected_candidates


@pytest.mark.parametrize(
    "choices, length_boundary, expected_candidates",
    [
        # Fixed-length list from explicit choices
        ([10, 20], (1, 1), {(10,), (20,)}),
        # Fixed longer length with combinations
        ([10, 20], (2, 2), {(10, 10), (10, 20), (20, 10), (20, 20)}),
        # With no length_boundary, the code should default to length 1.
        ([10, 20], None, {(10,), (20,)}),
    ],
)
def test_list_with_choices_only(choices, length_boundary, expected_candidates):
    """
    Test list generation using explicit choices with length constraints.
    Validates:
    - Choice-based list generation
    - Length boundary enforcement
    - Default length handling when no boundary specified
    - Combination generation for multi-element lists
    """

    @dataclass
    class ListChoicesConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        b: list[int] = field(
            default_factory=lambda: [10, 20, 30], metadata={"tunable": True}
        )

    tm = TuningMetadata(ListChoicesConfig)
    tm.set_metadata(
        {"b": FieldMetadata(choices=choices, length_boundary=length_boundary)}
    )

    config = ListChoicesConfig(tuning_metadata=tm)
    candidates = list(generate_all_configurations(config))

    result = {tuple(c.b) for c in candidates}

    assert result == expected_candidates


def test_list_no_range_no_choices():
    """
    Test list generation when no range or choices are specified.
    Validates:
    - Fallback to default list values as choices
    """

    @dataclass
    class EmptyListConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        b: list[int] = field(
            default_factory=lambda: [10, 20, 30], metadata={"tunable": True}
        )

    tm = TuningMetadata(EmptyListConfig)
    # Use b's default as choices
    tm.set_metadata({"b": FieldMetadata(length_boundary=(1, 2))})

    config = EmptyListConfig(tuning_metadata=tm)
    candidates = list(generate_all_configurations(config))

    # Expected candidates: [[10], [20]]
    result = {tuple(c.b) for c in candidates}

    # Verify all combinations of lengths 1 and 2 using default values
    assert result == {
        (10,),
        (20,),
        (30,),
        (10, 10),
        (10, 20),
        (10, 30),
        (20, 10),
        (20, 20),
        (20, 30),
        (30, 10),
        (30, 20),
        (30, 30),
    }


def test_nested_config():
    """
    Test nested dataclass configuration generation.
    Validates:
    - Recursive configuration generation
    - Proper metadata propagation to nested classes
    - Independent tuning of nested components
    """

    @dataclass
    class InnerConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        a: int = field(default=2, metadata={"tunable": True})

    @dataclass
    class OuterConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        inner: InnerConfig = None

    # Configure inner class metadata
    inner_tm = TuningMetadata(InnerConfig)
    inner_tm.set_metadata({"a": FieldMetadata(value_range=(1, 3, 1))})
    inner = InnerConfig(tuning_metadata=inner_tm)

    # Outer config without specific tuning
    outer_tm = TuningMetadata(OuterConfig)
    outer = OuterConfig(tuning_metadata=outer_tm, inner=inner)

    candidates = list(generate_all_configurations(outer))

    # Expect 3 candidate OuterConfigs (based on inner.a: 1,2,3)
    assert len(candidates) == 3
    for c in candidates:
        assert c.inner.a in [1, 2, 3]
