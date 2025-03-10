import numpy as np

from dataclasses import dataclass, fields, is_dataclass, asdict
from typing import Any, Optional, Union, Type, get_args, get_origin, get_type_hints
from itertools import product


@dataclass
class FieldMetadata:
    choices: Optional[list[Any]] = None
    value_range: Optional[
        Union[tuple[int, int, int], tuple[float, float, float]]
    ] = None
    length_boundary: Optional[tuple[int, int]] = None


class TuningMetadata:
    def __init__(self, dataclass_cls: Type):
        self.__metadata = {f.name: FieldMetadata() for f in fields(dataclass_cls)}
        self.__metadata.pop("tuning_metadata", None)

    def get(self, name, default):
        return self.__metadata.get(name, default)

    def set_metadata(self, metadata: Optional[dict]):
        if metadata is None:
            return

        for k, v in self.__metadata.items():
            self.__metadata[k] = metadata.get(k, v)


def _is_union_list_type(field_type: Any) -> Optional[Any]:
    """
    Checks if a field type is `Union[X, list[X]]`.

    Args:
        field_type: The annotated type of a field.

    Returns:
        The base type X if the field is Union[X, list[X]], otherwise None.
    """
    if get_origin(field_type) is Union:
        args = get_args(field_type)
        if len(args) == 2 and any(get_origin(arg) is list for arg in args):
            if get_origin(args[0]) is list:
                base = get_args(args[0])[0]
            else:
                base = get_args(args[1])[0]
            return base
    return None


def generate_all_configurations(config_instance: Any):
    """
    Recursively generate all candidate configurations for a dataclass instance by
    exhaustively exploring tunable fields according to their metadata.

    Field handling rules:
      - For `Union[X, list[X]]` with a provided length_boundary:
          Generate candidate lists of allowed lengths using `choices` (for `str`) or
          `value_range` (for `int`). Without a length_boundary, treat the field as scalar `X`.
      - For list fields:
          Generate candidate lists using either `choices` (e.g. for `list[str]`) or
          `value_range` (for `list[int/float]`). If a length_boundary is provided,
          generate lists for all allowed lengths; otherwise, use the current length.
      - For scalar numeric fields (`int/float`):
          Generate candidate values using `value_range`.
      - For scalar string fields:
          Generate candidate values using `choices`.
      - For fields without tuning metadata, retain the current value.

    Yields:
        New instances of the dataclass for every combination of candidate values.
    """
    if not is_dataclass(config_instance) or not hasattr(
        config_instance, "tuning_metadata"
    ):
        yield config_instance
        return

    # Dictionary to store possible values for each field
    candidate_dict = {}

    type_hints = get_type_hints(config_instance.__class__)
    tuning_metadata: TuningMetadata = config_instance.tuning_metadata

    for f in fields(config_instance):
        # Skip the tuning_metadata field itself.
        if f.name in ("tuning_metadata"):
            continue

        value = getattr(config_instance, f.name)

        meta = tuning_metadata.get(f.name, None)
        meta = asdict(meta) if meta else meta

        ftype = type_hints.get(f.name)
        candidates = []

        # Recursively generate candidates if the field is a nested dataclass.
        if is_dataclass(value):
            candidates = list(generate_all_configurations(value))
        # If no metadata, retain the current value.
        elif not meta:
            candidates = [value]
        else:
            # Case 1: Field is Union[X, list[X]]
            union_elem = _is_union_list_type(ftype)
            if union_elem is not None:
                # Generate candidate values for type X.
                if union_elem is str:
                    possible_vals = meta["choices"] or (
                        value if type(value) is list else [value]
                    )
                elif union_elem in (int, float):
                    if not (0 < bool(meta["value_range"]) + bool(meta["choices"]) < 2):
                        # In this case we either can't choose or can't configure
                        candidate_dict[f.name] = [value]
                        continue

                    if meta["choices"]:
                        possible_vals = meta["choices"]
                    else:
                        min_val, max_val, step = meta["value_range"]
                        if union_elem is int:
                            possible_vals = list(range(min_val, max_val + 1, step))
                        else:
                            possible_vals = []
                            current = min_val
                            while current <= max_val:
                                possible_vals.append(current)
                                current += step
                else:
                    candidate_dict[f.name] = [value]
                    continue

                # Generate lists for every allowed length.
                length_boundary = meta["length_boundary"]
                if length_boundary:
                    min_len, max_len = length_boundary

                    for l in range(min_len, max_len + 1):
                        candidates.extend(
                            [list(combo) for combo in product(possible_vals, repeat=l)]
                        )
                # Without length_boundary, treat as a scalar.
                else:
                    candidates = possible_vals

            # Case 2: Field is a list (but not a Union)
            elif get_origin(ftype) is list:
                underlying_type = get_args(ftype)[0]

                if underlying_type is str:
                    possible_vals = meta["choices"] or value
                elif underlying_type in (int, float):
                    if not (0 < bool(meta["value_range"]) + bool(meta["choices"]) < 2):
                        # In this case we either can't choose or can't configure
                        candidate_dict[f.name] = [value]
                        continue

                    if meta["choices"]:
                        possible_vals = meta["choices"]
                    else:
                        min_val, max_val, step = meta["value_range"]
                        if union_elem is int:
                            possible_vals = list(range(min_val, max_val + 1, step))
                        else:
                            possible_vals = []
                            current = min_val
                            while current <= max_val:
                                possible_vals.append(current)
                                current += step

                else:
                    candidate_dict[f.name] = [value]
                    continue

                # Determine length boundaries; default to current length if not provided.
                length_boundary = meta["length_boundary"]
                if length_boundary:
                    min_len, max_len = length_boundary
                else:
                    min_len, max_len = (len(value), len(value))

                # Generate lists for every allowed length.
                for l in range(min_len, max_len + 1):
                    for combo in product(possible_vals, repeat=l):
                        candidates.append(list(combo))

            # Case 3: Scalar field (int, str, etc.)
            else:
                if ftype is str:
                    candidates = meta["choices"] or [value]
                elif ftype in (int, float):
                    if not (0 < bool(meta["value_range"]) + bool(meta["choices"]) < 2):
                        # In this case we either can't choose or can't configure
                        candidate_dict[f.name] = [value]
                        continue

                    if meta["choices"]:
                        candidates = meta["choices"]
                    else:
                        min_val, max_val, step = meta["value_range"]
                        if union_elem is int:
                            candidates = list(range(min_val, max_val + 1, step))
                        else:
                            current = min_val
                            while current <= max_val:
                                candidates.append(current)
                                current += step
                else:
                    candidate_dict[f.name] = [value]
                    continue

        candidate_dict[f.name] = candidates

    # Generate the Cartesian product of candidate values for all fields.
    keys = list(candidate_dict.keys())
    for comb in product(*(candidate_dict[k] for k in keys)):
        # Build a candidate instance from the product.
        yield type(config_instance)(**dict(zip(keys, comb)))
