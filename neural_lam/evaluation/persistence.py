"""Persistence-forecast baseline helpers."""

import torch


def persistence_forecast(latest_observation, n_leads):
    """Repeat the most recent observed state for each requested forecast lead.

    ``latest_observation`` has shape ``(batch, grid_cell, variable)`` and the
    result has shape ``(batch, lead, grid_cell, variable)``.
    """
    if not isinstance(latest_observation, torch.Tensor):
        latest_observation = torch.as_tensor(latest_observation)
    if latest_observation.ndim != 3:
        raise ValueError("latest_observation must have shape (batch, grid, variable)")
    if n_leads <= 0:
        raise ValueError("n_leads must be positive")
    return latest_observation.unsqueeze(1).expand(
        -1, n_leads, -1, -1
    )
