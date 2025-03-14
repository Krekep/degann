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
        ((1, 3, 1), {1, 2, 3}),
        ((5, 5, 1), {5}),
    ],
)
def test_int_range_config(value_range, expected_values):
    @dataclass
    class IntConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        a: int = 2

    tm = TuningMetadata(IntConfig)
    tm.set_metadata({"a": FieldMetadata(value_range=value_range)})

    int_config = IntConfig(tuning_metadata=tm)
    candidates = list(generate_all_configurations(int_config))

    result_values = {c.a for c in candidates}

    assert result_values == expected_values


@pytest.mark.parametrize(
    "value_range, expected_values",
    [
        ((0.0, 0.2, 0.1), {0.0, 0.1, 0.2}),
        ((0.0, 0.0, 0.5), {0.0}),
    ],
)
def test_float_range_config(value_range, expected_values):
    @dataclass
    class FloatConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        a: float = 2.0

    tm = TuningMetadata(FloatConfig)
    tm.set_metadata({"a": FieldMetadata(value_range=value_range)})

    float_config = FloatConfig(tuning_metadata=tm)
    candidates = list(generate_all_configurations(float_config))

    result_values = {c.a for c in candidates}

    assert result_values == expected_values


@pytest.mark.parametrize(
    "choices, expected_values",
    [
        (("A", "B", "C"), {"A", "B", "C"}),
        (None, {"default"}),
    ],
)
def test_choice_config(choices, expected_values):
    @dataclass
    class ChoiceConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        c: str = "default"

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
        (
            (0.0, 3.0, 1.0),  # [0.0, 1.0, 2.0, 3.0]
            (0.5, 1.5, 2.5),  # [0.5, 1.5, 2.5]
            {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0},
        )
    ],
)
def test_mixed_config(value_range, choices, expected_values):
    @dataclass
    class MixedConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        c: float = 5.0

    tm = TuningMetadata(MixedConfig)
    tm.set_metadata({"c": FieldMetadata(choices=choices, value_range=value_range)})

    config = MixedConfig(
        tuning_metadata=tm,
    )
    candidates = list(generate_all_configurations(config))

    result_values = {instance.c for instance in candidates}

    assert result_values == expected_values


def test_no_tuning_metadata():
    @dataclass
    class NoTuneConfig:
        a: int = 10
        b: str = "test"

    config = NoTuneConfig()
    candidates = list(generate_all_configurations(config))

    # Should return only one candidate: the original configuration.
    assert len(candidates) == 1

    assert candidates[0].a == 10
    assert candidates[0].b == "test"


def test_empty_metadata():
    @dataclass
    class EmptyMetadataConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        a: int = 10
        b: str = "test"

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
                "a": FieldMetadata(value_range=(1, 2, 1)),
                "b": FieldMetadata(choices=[10, 20]),
                "c": FieldMetadata(choices=["x", "y"]),
            },
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
    @dataclass
    class MixedConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        a: int = 0
        b: list[int] = field(default_factory=lambda: [10])
        c: str = "x"

    tm = TuningMetadata(MixedConfig)
    tm.set_metadata(metadata_dict)

    config = MixedConfig(tuning_metadata=tm, a=0, b=[10], c="x")
    candidates = list(generate_all_configurations(config))

    generated_configs = {(cfg.a, tuple(cfg.b), cfg.c) for cfg in candidates}

    assert generated_configs == configurations


@pytest.mark.parametrize(
    "value_range, length_boundary, expected_candidates",
    [
        # value_range (10,20,10) yields [10,20]. With length_boundary (1,1): two candidates.
        ((10, 20, 10), (1, 1), {(10,), (20,)}),
        # With length_boundary (1,2): six candidates.
        ((10, 20, 10), (1, 2), {(10, 10), (10, 20), (20, 10), (20, 20), (10,), (20,)}),
    ],
)
def test_list_with_value_range_only(value_range, length_boundary, expected_candidates):
    @dataclass
    class ListRangeConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        b: list[int] = field(default_factory=lambda: [10])

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
        # With explicit length_boundary: length 1 should yield [[10],[20]].
        ([10, 20], (1, 1), {(10,), (20,)}),
        # With explicit length_boundary: length 2 should yield [[10,10],[10,20],[20,10],[20,20]].
        ([10, 20], (2, 2), {(10, 10), (10, 20), (20, 10), (20, 20)}),
        # With no length_boundary, the code should default to length 1.
        ([10, 20], None, {(10,), (20,)}),
    ],
)
def test_list_with_choices_only(choices, length_boundary, expected_candidates):
    @dataclass
    class ListChoicesConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        b: list[int] = field(default_factory=lambda: [10, 20, 30])

    tm = TuningMetadata(ListChoicesConfig)
    tm.set_metadata(
        {"b": FieldMetadata(choices=choices, length_boundary=length_boundary)}
    )

    config = ListChoicesConfig(tuning_metadata=tm)
    candidates = list(generate_all_configurations(config))

    result = {tuple(c.b) for c in candidates}

    assert result == expected_candidates


def test_empty_list_field():
    @dataclass
    class EmptyListConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        b: list[int] = field(default_factory=list)

    tm = TuningMetadata(EmptyListConfig)
    # Even if the field is initially empty, we want to generate candidates of length 1.
    tm.set_metadata({"b": FieldMetadata(choices=[10, 20])})

    config = EmptyListConfig(tuning_metadata=tm)
    candidates = list(generate_all_configurations(config))

    # Expected candidates: [[10], [20]]
    result = {tuple(c.b) for c in candidates}

    assert result == {(10,), (20,)}


def test_list_no_range_no_choices():
    @dataclass
    class EmptyListConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        b: list[int] = field(default_factory=lambda: [10, 20, 30])

    tm = TuningMetadata(EmptyListConfig)
    # Use b's default as choices
    tm.set_metadata({"b": FieldMetadata(length_boundary=(1, 2))})

    config = EmptyListConfig(tuning_metadata=tm)
    candidates = list(generate_all_configurations(config))

    # Expected candidates: [[10], [20]]
    result = {tuple(c.b) for c in candidates}

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
    @dataclass
    class InnerConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        a: int = 2

    @dataclass
    class OuterConfig:
        tuning_metadata: Optional[TuningMetadata] = None
        inner: InnerConfig = None

    inner_tm = TuningMetadata(InnerConfig)
    inner_tm.set_metadata({"a": FieldMetadata(value_range=(1, 3, 1))})
    inner = InnerConfig(tuning_metadata=inner_tm)

    outer_tm = TuningMetadata(OuterConfig)
    # Outer config does not need its own tuning for fields, so we leave it empty.
    outer = OuterConfig(tuning_metadata=outer_tm, inner=inner)

    candidates = list(generate_all_configurations(outer))

    # Expect 3 candidate OuterConfigs (based on inner.a: 1,2,3)
    assert len(candidates) == 3
    for c in candidates:
        assert c.inner.a in [1, 2, 3]
