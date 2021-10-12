"""Losses for training LGS Models."""
import elegy
import jax.numpy as jnp
import numpy as np


class KLDiv(elegy.Loss):
    """Kullback-Leibler Divergence for a Gaussian distribution over latent variables."""

    def __init__(self, beta: float = 1):
        super().__init__()
        self.beta = beta

    def call(self, y_pred: dict) -> np.ndarray:
        logvar = 2 * jnp.log(y_pred["latent_std"])
        mean = y_pred["latent_mean"]
        return (
            -self.beta
            / 2
            * jnp.mean((1 + logvar) - mean ** 2 - jnp.exp(logvar), axis=-1)
        )


class PhotometryMSE(elegy.losses.MeanSquaredError):
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


class ColorMSE(elegy.Loss):
    """Mean Square Error for the colors.

    Note I added a factor of 0.5 for the VAE definition, and I hardcoded
    an error of 0.01 mags for all photometry.
    """

    def __init__(self, ref_idx: int):
        super().__init__()
        self.ref_idx = ref_idx  # index of the reference band

    def call(self, x: np.ndarray, y_pred: dict) -> np.ndarray:
        mag_SE = (
            1
            / 0.01 ** 2
            * (
                x[..., self.ref_idx + 1]
                - y_pred["predicted_photometry"][..., self.ref_idx]
            )
            ** 2
        )

        color_SE = (
            1
            / (2 * 0.01 ** 2)
            * (
                jnp.diff(x[:, 1:], axis=-1)
                - jnp.diff(y_pred["predicted_photometry"], axis=-1)
            )
            ** 2
        ).sum(axis=-1)

        MSE = mag_SE + color_SE / y_pred["predicted_photometry"].shape[-1]
        return 0.5 * MSE


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


class SpectralLoss(elegy.Loss):
    """Calculates loss w.r.t. latent SEDs, assuming a fraction of true
    SEDs are known exactly.
    """

    def __init__(self, eta: float, frac: int):
        super().__init__()
        self.eta = eta
        self.frac = frac

    def call(self, y_true: np.ndarray, y_pred: dict) -> np.ndarray:
        N = int(self.frac * y_pred["amplitude"].shape[0])
        assert N > 0, "With this batch size, the given frac results in no spectra."
        true_sed = y_true[:N]
        pred_sed = y_pred["amplitude"][:N] + y_pred["sed_mag"][:N]
        return self.eta * jnp.mean((true_sed - pred_sed) ** 2)
