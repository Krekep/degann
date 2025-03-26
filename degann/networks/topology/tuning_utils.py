from dataclasses import dataclass, fields, is_dataclass, asdict, replace
from typing import Any, Optional, Union, Type, get_args, get_origin, get_type_hints
from itertools import product
from collections import defaultdict


@dataclass
class FieldMetadata:
    """
    A class representing metadata for tunable fields in a configuration.

    Attributes:
        choices (`Optional[list[Any]]`):
            A list of allowed values for the field.
        value_range (`Optional[Union[tuple[int, int, int], tuple[float, float, float]]]`):
            Defines a numeric range for the field in the format (min_value, max_value, step).
        length_boundary (`Optional[tuple[int, int]]`):
            Specifies the minimum and maximum allowed length for list-type fields.
            It follows the format (min_length, max_length).
    """

    choices: Optional[list[Any]] = None
    value_range: Optional[
        Union[tuple[int, int, int], tuple[float, float, float]]
    ] = None
    length_boundary: Optional[tuple[int, int]] = None


class TuningMetadata:
    """
    A class for managing tuning metadata associated with fields in a dataclass.

    This class initializes metadata for tunable fields in a given dataclass, allowing
    retrieval of metadata on a per-field basis.

    Attributes:
        __metadata (`dict`):
            A dictionary mapping field names to `FieldMetadata` instances,
            storing tuning-related metadata for each field.

    Methods:
        `get(name: str, default: Any) -> Any`:
            Retrieves the metadata for a given field.
            If the field is not found, returns the provided default value.

        `set_metadata(metadata: Optional[dict]) -> None`:
            Updates the metadata dictionary using the provided metadata.
            If `metadata` is None, the method does nothing.
            Only existing fields in `__metadata` are updated.
    """

    def __init__(self, dataclass_cls: Type):
        """
        Initializes the TuningMetadata instance.

        Args:
            dataclass_cls (`Type`): The dataclass for which metadata will be initialized.

        Constructor creates a metadata dictionary where each field in the
        dataclass is associated with a default `FieldMetadata` instance.
        The special `tuning_metadata` field (if present) is removed.
        """

        self.__metadata = {
            f.name: FieldMetadata()
            for f in fields(dataclass_cls)
            if f.metadata.get("tunable", False)
        }

    def get(self, name, default) -> Any:
        """
        Retrieves the metadata for a specific field.

        Args:
            name (`str`): The name of the field.
            default (`Any`): The default value to return if the field does not exist.

        Returns:
            `FieldMetadata` or `Any`: The metadata for the given field or the default value.
        """

        return self.__metadata.get(name, default)

    def set_metadata(self, metadata: Optional[dict]) -> None:
        """
        Updates the metadata dictionary with new values.

        Args:
            metadata (`Optional[dict]`): A dictionary containing new metadata values.

        If `metadata` is None, the function does nothing.
        Otherwise, it updates existing fields in `__metadata` with values from `metadata`,
        leaving fields unchanged if they are not present in the input dictionary.
        """

        if metadata is None:
            return

        for k, v in self.__metadata.items():
            self.__metadata[k] = metadata.get(k, v)


def generate_all_configurations(config_instance: Any):
    """
    Recursively generate all candidate configurations for a dataclass instance by
    exhaustively exploring tunable fields according to their metadata.

    Yields:
        New instances of the dataclass for every combination of candidate values.
    """
    if not (
        # Check if `config_instance` is dataclass instance
        is_dataclass(config_instance)
        and not isinstance(config_instance, type)
    ) or not hasattr(config_instance, "tuning_metadata"):
        # No tuning data? -> nothing to go through
        # Not a dataclass instance? -> not a config
        yield config_instance
        return

    # Dictionary to store possible values for each field
    candidate_dict = defaultdict(list)

    type_hints = get_type_hints(config_instance.__class__)
    tuning_metadata: TuningMetadata = config_instance.tuning_metadata

    for f in fields(config_instance):
        # Skip data that shouldn't be initialized.
        if not f.init:
            continue

        ftype = type_hints.get(f.name)
        value = getattr(config_instance, f.name)

        # Get metadata for field in dict format
        meta = tuning_metadata.get(f.name, None)
        meta = asdict(meta) if meta else meta

        # Recursively generate candidates if the field is a nested dataclass.
        if is_dataclass(value):
            candidate_dict[f.name] = list(generate_all_configurations(value))
            continue

        # meta -- dict from TuningMetadata and f.metadata -- dataclass wide dict
        if not meta or not f.metadata.get("tunable", False):
            # No metadata -> nothing to go through
            candidate_dict[f.name] = [value]
            continue

        # Determine base type and listability
        if get_origin(ftype) is list:
            base_type = get_args(ftype)[0]
            is_listable = True
        else:
            base_type = ftype
            is_listable = False

        # Generate possible values for the base type
        possible_vals: list[Any] = []

        vr = meta.get("value_range")
        ch = meta.get("choices")

        if base_type in (int, float) and vr is not None:
            # Extend `possible_vals` with values in range
            min_val, max_val, step = vr
            if base_type is int:
                possible_vals.extend(range(min_val, max_val + 1, step))
            else:
                current = min_val
                while current <= max_val:
                    possible_vals.append(current)
                    current += step

        if ch is not None:
            # Extend `possible_vals` with values from `choices`
            possible_vals.extend(ch)

        if not possible_vals:
            if isinstance(value, list):
                # Use values in list as possible ones
                possible_vals = value
            else:
                possible_vals = [value]

        # Generate list candidates if applicable
        if not is_listable:
            candidate_dict[f.name] = possible_vals
            continue

        length_boundary = meta.get("length_boundary")
        if length_boundary is None:
            # Generate list with length 1
            candidate_dict[f.name] = [[val] for val in possible_vals]
            continue

        # Generate lists with length in provided boundary
        min_len, max_len = length_boundary
        for l in range(min_len, max_len + 1):
            candidate_dict[f.name].extend(
                [list(c) for c in product(possible_vals, repeat=l)]
            )

    # Generate the Cartesian product of candidate values for all fields.
    keys = list(candidate_dict.keys())
    for comb in product(*(candidate_dict[k] for k in keys)):
        # Build a candidate instance from the product.
        yield replace(config_instance, **dict(zip(keys, comb)))
