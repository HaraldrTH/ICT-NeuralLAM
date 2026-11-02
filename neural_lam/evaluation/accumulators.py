"""Streaming accumulators for lead-time and variable-level forecast metrics."""

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


def _as_numpy(value, name):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise TypeError(f"{name} must contain numeric values")
    return array


class StreamingMetricAccumulator:
    """Accumulate physical-unit error statistics without retaining forecasts.

    Predictions and targets use shape ``(batch, lead, grid_cell, variable)``.
    Grid weights, when supplied, are applied to metric averages; ``sample_count``
    always reports the number of valid batch/grid-cell observations.
    """

    def __init__(
        self,
        variable_names: Sequence[str],
        lead_hours: Sequence[int],
        training_std: Optional[Sequence[float]] = None,
        variable_units: Optional[Sequence[str]] = None,
    ):
        self.variable_names = tuple(variable_names)
        self.lead_hours = tuple(int(value) for value in lead_hours)
        if not self.variable_names:
            raise ValueError("variable_names must not be empty")
        if not self.lead_hours or any(value <= 0 for value in self.lead_hours):
            raise ValueError("lead_hours must contain positive values")
        if len(set(self.lead_hours)) != len(self.lead_hours):
            raise ValueError("lead_hours must not contain duplicates")

        self.training_std = None
        if training_std is not None:
            std = np.asarray(training_std, dtype=np.float64)
            if std.shape != (len(self.variable_names),):
                raise ValueError("training_std must have one value per variable")
            if not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError("training_std values must be finite and positive")
            self.training_std = std

        self.variable_units = tuple(variable_units or ("",) * len(self.variable_names))
        if len(self.variable_units) != len(self.variable_names):
            raise ValueError("variable_units must have one value per variable")

        shape = (len(self.lead_hours), len(self.variable_names))
        self._absolute_error_sum = np.zeros(shape, dtype=np.float64)
        self._squared_error_sum = np.zeros(shape, dtype=np.float64)
        self._weight_sum = np.zeros(shape, dtype=np.float64)
        self._sample_count = np.zeros(shape, dtype=np.int64)
        self._persistence_absolute_error_sum = np.zeros(shape, dtype=np.float64)
        self._persistence_squared_error_sum = np.zeros(shape, dtype=np.float64)
        self._has_persistence = None
        self._n_batches = 0

    def update(
        self,
        prediction,
        target,
        persistence_prediction=None,
        valid_mask=None,
        grid_weights=None,
    ):
        """Add one forecast batch; inputs must be in physical units."""
        prediction = _as_numpy(prediction, "prediction").astype(np.float64, copy=False)
        target = _as_numpy(target, "target").astype(np.float64, copy=False)
        expected_dims = 4
        if prediction.ndim != expected_dims or target.ndim != expected_dims:
            raise ValueError("prediction and target must have shape (batch, lead, grid, variable)")
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes must match")
        if prediction.shape[1] != len(self.lead_hours):
            raise ValueError("forecast lead dimension does not match lead_hours")
        if prediction.shape[-1] != len(self.variable_names):
            raise ValueError("forecast variable dimension does not match variable_names")

        if valid_mask is None:
            mask = np.ones(prediction.shape, dtype=bool)
        else:
            mask = _as_numpy(valid_mask, "valid_mask").astype(bool, copy=False)
            try:
                mask = np.broadcast_to(mask, prediction.shape)
            except ValueError as exc:
                raise ValueError("valid_mask is not broadcastable to forecast shape") from exc

        if not np.isfinite(prediction[mask]).all() or not np.isfinite(target[mask]).all():
            raise ValueError("valid prediction and target values must be finite")

        if grid_weights is None:
            weights = np.ones((1, 1, prediction.shape[2], 1), dtype=np.float64)
        else:
            weights = _as_numpy(grid_weights, "grid_weights").astype(
                np.float64, copy=False
            )
            if weights.shape != (prediction.shape[2],):
                raise ValueError("grid_weights must have one value per grid cell")
            if not np.isfinite(weights).all() or np.any(weights < 0):
                raise ValueError("grid_weights must be finite and non-negative")
            weights = weights.reshape(1, 1, -1, 1)

        has_persistence = persistence_prediction is not None
        if self._has_persistence is not None and self._has_persistence != has_persistence:
            raise ValueError("persistence predictions must be supplied consistently")
        self._has_persistence = has_persistence

        errors = prediction - target
        weighted_mask = mask * weights
        axes = (0, 2)
        self._absolute_error_sum += np.sum(
            np.where(mask, np.abs(errors), 0.0) * weights, axis=axes
        )
        self._squared_error_sum += np.sum(
            np.where(mask, errors**2, 0.0) * weights, axis=axes
        )
        self._weight_sum += np.sum(weighted_mask, axis=axes)
        self._sample_count += np.sum(mask, axis=axes)
        self._n_batches += 1

        if persistence_prediction is not None:
            persistence_prediction = _as_numpy(
                persistence_prediction, "persistence_prediction"
            ).astype(np.float64, copy=False)
            if persistence_prediction.shape != prediction.shape:
                raise ValueError("persistence_prediction shape must match prediction")
            if not np.isfinite(persistence_prediction[mask]).all():
                raise ValueError("valid persistence predictions must be finite")
            persistence_errors = persistence_prediction - target
            self._persistence_absolute_error_sum += np.sum(
                np.where(mask, np.abs(persistence_errors), 0.0) * weights,
                axis=axes,
            )
            self._persistence_squared_error_sum += np.sum(
                np.where(mask, persistence_errors**2, 0.0) * weights,
                axis=axes,
            )

    @staticmethod
    def _divide(numerator, denominator):
        result = np.full(numerator.shape, np.nan, dtype=np.float64)
        np.divide(numerator, denominator, out=result, where=denominator > 0)
        return result

    def compute(self) -> Dict[str, np.ndarray]:
        """Return arrays indexed by ``(lead, variable)``; undefined values are NaN."""
        if self._n_batches == 0:
            raise ValueError("Cannot compute metrics before adding a forecast batch")
        mae = self._divide(self._absolute_error_sum, self._weight_sum)
        rmse = np.sqrt(self._divide(self._squared_error_sum, self._weight_sum))
        results = {"mae": mae, "rmse": rmse}
        if self.training_std is not None:
            results["normalized_mae"] = mae / self.training_std[None, :]
            results["normalized_rmse"] = rmse / self.training_std[None, :]
        if self._has_persistence:
            persistence_mae = self._divide(
                self._persistence_absolute_error_sum, self._weight_sum
            )
            persistence_rmse = np.sqrt(
                self._divide(self._persistence_squared_error_sum, self._weight_sum)
            )
            results["persistence_mae"] = persistence_mae
            results["persistence_rmse"] = persistence_rmse
            results["mae_skill_vs_persistence"] = 1.0 - self._divide(
                mae, persistence_mae
            )
            results["rmse_skill_vs_persistence"] = 1.0 - self._divide(
                rmse, persistence_rmse
            )
        return results

    def to_records(
        self,
        model_id: str,
        checkpoint_id: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
        metric_names: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, object]]:
        """Produce one tidy record per metric, lead time, and variable."""
        metric_values = self.compute()
        metadata = metadata or {}
        if metric_names is None:
            selected_metrics = tuple(metric_values)
        else:
            selected_metrics = tuple(metric_names)
            unsupported_metrics = set(selected_metrics) - set(metric_values)
            if unsupported_metrics:
                raise ValueError(
                    "Requested metrics are unavailable: "
                    + ", ".join(sorted(unsupported_metrics))
                )
        records = []
        for lead_index, lead_hours in enumerate(self.lead_hours):
            for variable_index, variable_name in enumerate(self.variable_names):
                for metric_name in selected_metrics:
                    values = metric_values[metric_name]
                    value = float(values[lead_index, variable_index])
                    record = {
                        "model_id": model_id,
                        "checkpoint_id": checkpoint_id,
                        "lead_hours": lead_hours,
                        "variable": variable_name,
                        "unit": (self.variable_units[variable_index]
                                if metric_name in {"mae", "rmse",
                                                   "persistence_mae",
                                                   "persistence_rmse"} else "1"),
                        "metric": metric_name,
                        "value": value if np.isfinite(value) else None,
                        "sample_count": int(self._sample_count[lead_index, variable_index]),
                        "weight_sum": float(self._weight_sum[lead_index, variable_index]),
                    }
                    record.update(metadata)
                    records.append(record)
        return records
