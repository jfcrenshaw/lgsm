"""
Module that translates a latent SED to observed photometry.
"""
from typing import Sequence

import elegy
import jax.numpy as jnp
import numpy as np
import sncosmo
from jax import vmap
from sncosmo.constants import HC_ERG_AA

from lgsm.sed_utils import mag_to_flambda


class PhysicsLayer(elegy.Module):
    """A physics layer that calculates photometry from an SED in AB magnitude.

    1. Calculating fluxes from F_lambda
    CCDs are photon counters, so we must convert from energy (ergs) to photon
    counts. We can do this via dividing by the energy/photon:
    energy/photon = hc / lambda.
    To calculate the flux through a bandpass, we convolve this with the
    dimensionless filter response function, T(lambda):
    flux = 1/hc * Integral[dlambda lambda T(lambda) F_lambda(lambda)].
    The units of this flux are photon/s/cm^2.

    2. Converting this flux back to an AB magnitude.
    We need to divide this flux by the flux of an object with magnitude zero
    in the corresponding band (i.e. the integrated flux of the 3631 Jy SED
    through the corresponding band). We call this the zero point (zp) flux.
    Thus, the AB magnitude through the band is m_AB = -2.5 * log10(flux/zp).
    """

    def __init__(
        self,
        sed_wave: np.ndarray,
        sed_unit: str,
        bandpasses: Sequence[str],
        band_oversampling: int,
    ):
        super().__init__()

        # make sure that sed_wave has even sampling
        dlambda = np.diff(sed_wave)
        assert jnp.all(
            jnp.allclose(dlambda, dlambda[0])
        ), "sed_wave must be an evenly-spaced grid."

        # save config
        self.sed_wave = sed_wave
        self.sed_unit = sed_unit
        self.bandpasses = bandpasses
        self.band_oversampling = band_oversampling

        # setup functions to handle sed units
        if self.sed_unit == "mag":
            self.scale_sed = lambda sed, amplitude: amplitude + sed
            self.convert_sed_units = lambda sed: mag_to_flambda(sed, self.sed_wave)
        else:  # sed_unit == "flambda"
            self.scale_sed = lambda sed, amplitude: amplitude * sed
            self.convert_sed_units = lambda sed: sed

        # precompute the  weights for flux integration
        self.band_wave, self.band_weights = self._get_band_weights()

    def _get_band_weights(self):
        """Precompute the weights for flux integration so they can be reused
        over and over again!
        By our definition,
        integrated flux = (F_lambda * weights).sum().
        Note that you can optionally oversample the bandpasses. This is so they
        have a finer wavelength sampling than the SED itself. This can reduce
        computational errors in calculating the integrated flux.
        """

        # check if oversampling bands
        assert (
            self.band_oversampling % 2 == 1
        ), "band_oversampling must be an odd integer."
        pad = (self.band_oversampling - 1) // 2

        # wavelength grid for bandpass is essentially the same as wave,
        # potentially with oversampling
        dwave = self.sed_wave[1] - self.sed_wave[0]
        band_dwave = dwave / self.band_oversampling
        band_wave = jnp.arange(
            self.sed_wave.min() - band_dwave * pad,
            self.sed_wave.max() + dwave + band_dwave * pad,
            band_dwave,
        )

        # set the overall flux normalization
        norm = 1 / HC_ERG_AA  # photons / erg / AA

        # calculate the weights for each bandpass
        band_weights = []
        for band in self.bandpasses:

            # get the integrated zero point flux for the band
            zero_point = sncosmo.get_magsystem("ab").zpbandflux(band)

            # get the filter response function
            response = sncosmo.get_bandpass(band)(band_wave)

            # note we don't have to worry about normalizing the response
            # function because the same normalization appears in zero_point,
            # which cancels when we divide by zero_point.

            # calculate the weights needed for flux integration
            weights = norm / zero_point * response * band_wave * band_dwave

            # convolve the weights so that every point in conv_weights is
            # the sum of the adjacent N=oversampling elements in weights.
            conv_weights = jnp.convolve(
                weights, jnp.ones(self.band_oversampling), mode="valid"
            )
            band_weights.append(conv_weights)

        # return the band wavelength grid and the convolved weights
        band_wave = band_wave[pad : -pad or None]
        band_weights = jnp.array(band_weights)
        return band_wave, band_weights

    def _calc_mag_single_sed(self, sed_flambda, redshift, band_weights):

        # instead of redshifting the SED, we can blueshift the filters
        # (I'm not sure why, but doing so makes this simple calculation
        # far more accurate than redshifting SED - at least when comparing
        # to calculations with sncosmo...)
        band_weights = jnp.interp(
            self.sed_wave,
            self.band_wave / (1 + redshift),
            band_weights,
            left=0,
            right=0,
        )

        # calculate the integrated flux through the band
        flux = (sed_flambda * band_weights).sum()
        # convert to AB magnitude
        mag = -2.5 * jnp.log10(flux)

        return mag

    def _calc_mags_single_sed(self, sed_flambda, redshift):
        """Vectorize _calc_mag_single_sed to return mags for all bands."""
        return vmap(
            lambda band_weights: self._calc_mag_single_sed(
                sed_flambda, redshift, band_weights
            )
        )(self.band_weights)

    def _calc_mags_multiple_seds(self, sed_flambda, redshift):
        """Vectorize _calc_mags_single_sed to return mags for multiple SEDs."""
        return vmap(self._calc_mags_single_sed)(sed_flambda, redshift)

    def call(
        self,
        sed: np.ndarray,
        amplitude: np.ndarray,
        redshift: np.ndarray,
    ) -> dict:
        """Calculate fluxes for the SEDs at the appropriate amplitude and redshifts."""

        sed_scaled = self.scale_sed(sed, amplitude)
        sed_flambda = self.convert_sed_units(sed_scaled)

        mags = self._calc_mags_multiple_seds(sed_flambda, redshift)

        return {"predicted_photometry": mags}
