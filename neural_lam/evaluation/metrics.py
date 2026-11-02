"""Metric helpers and canonical variable groupings."""

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch


VARIABLE_GROUPS = {
    "surface_temperature": ("t2m", "d2m", "sst"),
    "surface_wind_components": ("u10", "v10"),
    "pressure": ("msl",),
    "upper_air_wind_components": tuple(
        f"{component}_{level}"
        for component in ("u", "v")
        for level in (500, 700, 850, 925)
    ),
    "moisture": ("q_500", "q_700", "q_850", "q_925", "tp"),
}


def derive_10m_wind_speed(values, variable_names: Sequence[str]):
    """Return wind speed from physical-unit u10/v10 values along the last axis."""
    names = tuple(variable_names)
    try:
        u_index, v_index = names.index("u10"), names.index("v10")
    except ValueError as exc:
        raise ValueError("variable_names must include both u10 and v10") from exc
    if isinstance(values, torch.Tensor):
        return torch.sqrt(values[..., u_index] ** 2 + values[..., v_index] ** 2)
    array = np.asarray(values)
    return np.hypot(array[..., u_index], array[..., v_index])


def resolve_variable_groups(
    variable_names: Sequence[str],
    groups: Mapping[str, Sequence[str]] = VARIABLE_GROUPS,
) -> Dict[str, Tuple[int, ...]]:
    """Resolve configured group names to the variables present in a dataset."""
    indices = {name: index for index, name in enumerate(variable_names)}
    return {
        group_name: tuple(indices[name] for name in names if name in indices)
        for group_name, names in groups.items()
    }
