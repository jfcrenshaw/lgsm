"""Module that wraps StandardScaler, VAE, and PhysicsLayer into a full LGSM Model."""
from typing import Sequence

import elegy
import numpy as np

from lgsm.physics_layer import PhysicsLayer
from lgsm.sed_utils import setup_wave_grid
from lgsm.vae import VAE


class StandardScaler(elegy.Module):
    """Standard Scaler that ensures input dimensions have mean zero and unit variance."""

    def __init__(self, input_mean: np.ndarray, input_std: np.ndarray):
        super().__init__()
        self.input_mean = input_mean
        self.input_std = input_std

    def call(self, inputs):
        return (inputs - self.input_mean) / self.input_std


class LGSModel(elegy.Module):
    """A Latent Galaxy SED Model (LGSM)."""

    def __init__(
        self,
        # StandardScaler settings
        input_mean: np.ndarray,
        input_std: np.ndarray,
        # Encoder settings
        encoder_layers: Sequence[int],
        intrinsic_latent_size: int,
        # Decoder settings
        decoder_layers: Sequence[int],
        wave_min: float,
        wave_max: float,
        wave_bins: int,
        normalize_at: float,
        sed_unit: str,
        # VAE settings
        batch_norm: bool,
        # PhysicsLayer settings
        bandpasses: Sequence[str],
        band_oversampling: int,
    ):
        super().__init__()

        # instantiate the Standard Scaler
        self.standard_scaler = StandardScaler(input_mean, input_std)

        # setup the sed wavelength grid
        sed_wave = setup_wave_grid(wave_min, wave_max, wave_bins)

        # instantiate the VAE
        self.vae = VAE(
            encoder_layers=encoder_layers,
            intrinsic_latent_size=intrinsic_latent_size,
            decoder_layers=decoder_layers,
            sed_wave=sed_wave,
            normalize_at=normalize_at,
            sed_unit=sed_unit,
            batch_norm=batch_norm,
        )

        # instantiate the Physics Layer
        self.physics_layer = PhysicsLayer(
            sed_wave=sed_wave,
            sed_unit=sed_unit,
            bandpasses=bandpasses,
            band_oversampling=band_oversampling,
        )

    def call(self, inputs: np.ndarray) -> dict:

        # pull out the redshifts
        redshift = inputs[:, 0, None]

        # standard scale VAE inputs
        inputs = self.standard_scaler(inputs)

        # pass the scaled inputs through the VAE
        vae_outputs = self.vae(inputs)

        # predict photometry for the intrinsic SEDs
        predicted_photometry = self.physics_layer(
            sed=vae_outputs[f"sed_{self.physics_layer.sed_unit}"],
            amplitude=vae_outputs["amplitude"],
            redshift=redshift,
        )

        # return all the results together
        return {
            **vae_outputs,
            **predicted_photometry,
            "redshift": redshift,
        }
