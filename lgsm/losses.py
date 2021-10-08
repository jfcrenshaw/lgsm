"""Losses for training LGS Models."""
import elegy
import jax.numpy as jnp
import numpy as np


class MSE(elegy.losses.MeanSquaredError):
    """Mean Square Error for photometry."""

    def call(self, x: np.ndarray, y_pred: dict, **kwargs) -> np.ndarray:
        return super().call(y_true=x[:, 1:], y_pred=y_pred["predicted_photometry"])


class SlopeLoss(elegy.Loss):
    """Penalty on differences in neighboring bins."""

    def __init__(self, eta: float, **kwargs):
        super().__init__()
        self.eta = eta

    def call(self, y_pred: dict) -> np.ndarray:
        return self.eta * jnp.mean(jnp.diff(y_pred["sed_mag"], axis=-1) ** 2)
