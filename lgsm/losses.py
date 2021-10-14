"""Losses for training LGS Models."""
import elegy
import jax.numpy as jnp
import numpy as np
from jax import random


class KLDiv(elegy.Loss):
    """Kullback-Leibler Divergence for a Gaussian distribution over latent variables."""

    def __init__(self, alpha: float = 0):
        super().__init__()
        self.alpha = alpha

    def call(self, y_pred: dict) -> np.ndarray:
        logvar = 2 * jnp.log(y_pred["latent_std"])
        mean = y_pred["latent_mean"]
        return (
            -(1 - self.alpha)
            / 2
            * jnp.mean((1 + logvar) - mean ** 2 - jnp.exp(logvar), axis=-1)
        )


class MMD(elegy.Loss):
    """Maximum-Mean Discrepancy, calculated using the method found in this tutorial
    https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
    """

    def __init__(self, alpha: float = 0, beta: float = 1e3, nsamples: int = 200):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.nsamples = nsamples

    @staticmethod
    def kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes mean of the Gaussian kernel matrix
        """
        # set the hyperparam to dim / 2
        dim = x.shape[1]
        sigma_sq = float(dim / 2)

        # compute the matrix of square distances
        sq_dist_matrix = jnp.sum((x[:, None, ...] - y[None, :, ...]) ** 2, axis=-1)

        # compute and return mean of the kernel matrix
        return jnp.exp(-sq_dist_matrix / (2 * sigma_sq)).mean()

    def call(self, y_pred: dict) -> np.ndarray:
        # get the latent samples from the training set
        train_latent_samples = y_pred["intrinsic_latents"]

        # generate latent samples from the true latent distribution
        PRNGKey = random.PRNGKey(train_latent_samples[0, 0].astype(int))
        true_latent_samples = random.normal(
            PRNGKey, shape=[self.nsamples, train_latent_samples.shape[1]]
        )

        # calculate MMD
        E_xx = self.kernel(train_latent_samples, train_latent_samples)
        E_yy = self.kernel(true_latent_samples, true_latent_samples)
        E_xy = self.kernel(train_latent_samples, true_latent_samples)
        mmd = E_xx + E_yy - 2 * E_xy
        return (self.alpha + self.beta - 1) * mmd


class PhotometryMSE(elegy.losses.MeanSquaredError):
    """Mean Square Error for photometry.

    Note I added a factor of 0.5 for the VAE definition, and 1/0.05^2 to hardcode
    an error of 0.05 mags for all photometry. Will change this later.
    """

    def call(self, x: np.ndarray, y_pred: dict) -> np.ndarray:
        return (
            0.5
            / 0.05 ** 2
            * super().call(y_true=x[:, 1:], y_pred=y_pred["predicted_photometry"])
        )


class ColorMSE(elegy.Loss):
    """Mean Square Error for the colors.

    Note I added a factor of 0.5 for the VAE definition, and I hardcoded
    an error of 0.05 mags for all photometry.
    """

    def __init__(self, ref_idx: int):
        super().__init__()
        self.ref_idx = ref_idx  # index of the reference band

    def call(self, x: np.ndarray, y_pred: dict) -> np.ndarray:
        mag_SE = (
            1
            / 0.05 ** 2
            * (
                x[..., self.ref_idx + 1]
                - y_pred["predicted_photometry"][..., self.ref_idx]
            )
            ** 2
        )

        color_SE = (
            1
            / (2 * 0.05 ** 2)
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

    I also hardcoded an error of 0.05 mags here
    """

    def __init__(self, eta: float):
        super().__init__()
        self.eta = eta

    def call(self, y_pred: dict) -> np.ndarray:
        return (
            1
            / 0.05 ** 2
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
