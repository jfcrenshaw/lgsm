"""Module that wraps StandardScaler, VAE, and PhysicsLayer into a full LGSM Model."""
from typing import Sequence

import elegy
import numpy as np

from .physics_layer import PhysicsLayer
from .vae import VAE


class StandardScaler(elegy.Module):
    """Standard Scaler that ensures input dimensions have mean zero and unit variance."""

    def __init__(self, input_mean: np.ndarray, input_std: np.ndarray):
        super().__init__()
        self.input_mean = input_mean
        self.input_std = input_std

    def call(self, inputs, **kwargs):
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
        sed_min: float,
        sed_max: float,
        sed_bins: int,
        normalize_at: float,
        sed_unit: str,
        # VAE settings
        batch_norm: bool,
        # PhysicsLayer settings
        bandpasses: Sequence[str],
        band_oversampling: int,
    ):
        super().__init__()
        self.input_mean = input_mean
        self.input_std = input_std
        self.encoder_layers = encoder_layers
        self.intrinsic_latent_size = intrinsic_latent_size
        self.decoder_layers = decoder_layers
        self.sed_min = sed_min
        self.sed_max = sed_max
        self.sed_bins = sed_bins
        self.normalize_at = normalize_at
        self.sed_unit = sed_unit
        self.batch_norm = batch_norm
        self.bandpasses = bandpasses
        self.band_oversampling = band_oversampling

    def call(self, inputs: np.ndarray) -> dict:

        # pull out the redshifts
        redshift = inputs[:, 0, None]

        # standard scale VAE inputs
        inputs = StandardScaler(self.input_mean, self.input_std)(inputs)

        # pass the scaled inputs through the VAE
        vae_outputs = VAE(
            encoder_layers=self.encoder_layers,
            intrinsic_latent_size=self.intrinsic_latent_size,
            decoder_layers=self.decoder_layers,
            wave_min=self.wave_min,
            wave_max=self.wave_max,
            wave_bins=self.wave_bins,
            normalize_at=self.normalize_at,
            sed_unit=self.sed_unit,
            batch_norm=self.batch_norm,
        )(inputs)

        # predict photometry for the intrinsic SEDs
        predicted_photometry = PhysicsLayer(
            self.wave_min,
            self.wave_max,
            self.wave_bins,
            self.sed_unit,
            self.bandpasses,
            self.band_oversampling,
        )(
            sed=vae_outputs[f"sed_{self.sed_unit}"],
            amplitude=vae_outputs["amplitude"],
            redshift=redshift,
        )

        # return all the results together
        return {
            **vae_outputs,
            **predicted_photometry,
            "redshift": redshift,
        }
