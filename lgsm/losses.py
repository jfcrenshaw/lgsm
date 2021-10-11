"""Losses for training LGS Models."""
import elegy
import jax.numpy as jnp
import numpy as np


class KLDiv(elegy.Loss):
    """Kullback-Leibler Divergence for a Gaussian distribution over latent variables."""

    def call(self, y_pred: dict) -> np.ndarray:
        logvar = 2 * jnp.log(y_pred["latent_std"])
        mean = y_pred["latent_mean"]
        return -0.5 * jnp.mean((1 + logvar) - mean ** 2 - jnp.exp(logvar), axis=-1)


class MSE(elegy.losses.MeanSquaredError):
    """Mean Square Error for photometry.

    Note I added a factor of 0.5 for the VAE definition, and 1/0.01 to hardcode
    an error of 0.01 mags for all photometry. Will change this later.
    """

    def call(self, x: np.ndarray, y_pred: dict) -> np.ndarray:
        return (
            0.5
            / 0.01 ** 2
            * super().call(y_true=x[:, 1:], y_pred=y_pred["predicted_photometry"])
        )


class SlopeLoss(elegy.Loss):
    """Penalty on differences in neighboring bins.

    I also hardcoded an error of 0.01 mags here
    """

    def __init__(self, eta: float):
        super().__init__()
        self.eta = eta

    def call(self, y_pred: dict) -> np.ndarray:
        return (
            1
            / 0.01 ** 2
            * self.eta
            * jnp.mean(jnp.diff(y_pred["sed_mag"], axis=-1) ** 2)
        )
