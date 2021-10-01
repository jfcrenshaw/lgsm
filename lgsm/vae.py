"""Module that defines the VAE used to model intrinsic spectra."""

from typing import Sequence

import elegy
import jax
import jax.numpy as jnp
import numpy as np


class IdentityLayer:
    """Identity layer that replaces BatchNormalization when batch_norm == False"""

    # pylint: disable=R0903

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return inputs


class Encoder(elegy.Module):
    """
    Encoder that maps redshift, photometry, and photometric errors
    onto intrinsic and extrinsic latent variables.
    """

    def __init__(
        self,
        layers: Sequence[int],
        intrinsic_latent_size: int,
        batch_norm: bool,
    ):
        super().__init__()

        self.layers = layers
        self.intrinsic_latent_size = intrinsic_latent_size

        # I hard-code the size of the extrinsic latent variables so that
        # the user cannot change this. This is because changing the extrinsic
        # latent variables requires that you change the PhysicsLayer
        self.extrinsic_latent_size = 1

        # setup the NormLayer
        # pylint: disable=C0103
        if batch_norm:
            self.NormLayer = elegy.nn.BatchNormalization
        else:
            self.NormLayer = IdentityLayer

    def call(self, inputs: np.ndarray) -> dict:

        # pull out the redshift, which I will assume is the first column
        redshift = inputs[:, 0, None]

        # construct the first encoder layers
        for layer in self.layers:
            inputs = elegy.nn.Linear(layer)(inputs)
            inputs = self.NormLayer()(inputs)
            inputs = jax.nn.relu(inputs)
            self.add_summary("relu", jax.nn.relu, inputs)

        # calculate the intrinsic latent variables, which are drawn from
        # a normal distribution
        mean = elegy.nn.Linear(self.intrinsic_latent_size, name="linear_mean")(inputs)
        log_stds = elegy.nn.Linear(self.intrinsic_latent_size, name="linear_std")(
            inputs
        )
        stds = jnp.exp(log_stds)
        intrinsic_latents = mean + stds * jax.random.normal(self.next_key(), mean.shape)

        # calculate the extrinsic latent variables (not including the redshift,
        # which was pulled out of the inputs above)
        extrinsic_latents = elegy.nn.Linear(
            self.extrinsic_latent_size, name="linear_extrinsic"
        )(inputs)

        # decide which column is which
        # (note: right now this is useless as there is only one...)
        amplitude = extrinsic_latents[:, 0, None]

        return {
            "intrinsic_latents": intrinsic_latents,
            "amplitude": amplitude,
            "redshift": redshift,
        }


class Decoder(elegy.Module):
    """Decoder that maps intrinsic latents onto an SED in AB magnitude."""

    def __init__(
        self,
        layers: Sequence[int],
        sed_wave: np.ndarray,
        normalize_at: float,
        sed_unit: str,
        batch_norm: bool,
    ):
        super().__init__()

        self.layers = layers
        self.sed_wave = sed_wave
        self.normalize_at = normalize_at
        self.sed_unit = sed_unit

        # define the function to normalize the SEDs
        if sed_unit == "mag":
            self.normalize = lambda sed, norm: sed - norm
        else:
            self.normalize = lambda sed, norm: sed / norm

        # setup the NormLayer
        # pylint: disable=C0103
        if batch_norm:
            self.NormLayer = elegy.nn.BatchNormalization
        else:
            self.NormLayer = IdentityLayer

    def call(self, intrinsic_latents: np.ndarray) -> dict:

        # construct the first decoder layers
        for layer in self.layers:
            intrinsic_latents = elegy.nn.Linear(layer)(intrinsic_latents)
            intrinsic_latents = self.NormLayer()(intrinsic_latents)
            intrinsic_latents = jax.nn.relu(intrinsic_latents)
            self.add_summary("relu", jax.nn.relu, intrinsic_latents)

        # calculate the SEDs
        sed = elegy.nn.Linear(self.sed_wave.size)(intrinsic_latents)

        # normalize the SEDs
        norm = jax.vmap(lambda sed: jnp.interp(self.normalize_at, self.sed_wave, sed))(
            sed
        ).reshape(-1, 1)
        sed = self.normalize(sed, norm)

        # return the wavelength grid and the normalized sed!
        return {"sed_wave": self.sed_wave, f"sed_{self.sed_unit}": sed}


class VAE(elegy.Module):
    """VAE that produces intrinsic spectra for (redshift, photometry) sets."""

    def __init__(
        self,
        # Encoder settings
        encoder_layers: Sequence[int],
        intrinsic_latent_size: int,
        # Decoder settings
        decoder_layers: Sequence[int],
        sed_wave: np.ndarray,
        normalize_at: float,
        sed_unit: str,
        # global settings
        batch_norm: bool,
    ):
        super().__init__()

        # save the config
        self.encoder_layers = encoder_layers
        self.intrinsic_latent_size = intrinsic_latent_size
        self.decoder_layers = decoder_layers
        self.sed_wave = sed_wave
        self.normalize_at = normalize_at
        self.sed_unit = sed_unit
        self.batch_norm = batch_norm

        # instantiate the encoder
        self.encoder = Encoder(
            layers=encoder_layers,
            intrinsic_latent_size=intrinsic_latent_size,
            batch_norm=batch_norm,
        )

        # instantiate the decoder
        self.decoder = Decoder(
            layers=decoder_layers,
            sed_wave=sed_wave,
            normalize_at=normalize_at,
            sed_unit=sed_unit,
            batch_norm=batch_norm,
        )

    def call(self, inputs: np.ndarray) -> dict:

        # encode the photometry in the implicit/explicit latent space
        latents = self.encoder(inputs)

        # decode into an SED
        sed = self.decoder(latents["intrinsic_latents"])

        return {**latents, **sed}
