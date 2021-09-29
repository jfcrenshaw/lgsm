"""Utilities for converting sed units back and forth."""
import jax.numpy as jnp
import numpy as np


def mag_to_fnu(mag: np.ndarray) -> np.ndarray:
    """Convert AB magnitudes to F_nu in erg/s/Hz/cm^2.
    In the AB magnitude system, an object with flux density F_nu = 3631 Jansky
    has a magnitude of zero in all bands (note: 1 Jansky = 1e-23 erg/s/Hz/cm^2).
    Thus, to convert from AB magnitude to F_nu, we do the following:
    F_nu = (3631e-23 erg/s/Hz/cm^2) * 10^(m_AB / -2.5).
    """
    fnu = 3631e-23 * 10 ** (mag / -2.5)
    return fnu


def fnu_to_flambda(fnu: np.ndarray, wave: np.ndarray) -> np.ndarray:
    """Convert F_nu in erg/s/Hz/cm^2 to F_lambda in erg/s/AA/cm^2.
    To convert from F_nu to F_lambda, we use the formula
    F_lambda = c / lambda^2 * F_nu,
    resulting in an SED with units of erg/s/cm^2/AA.
    """
    flambda = 2.998e18 / wave ** 2 * fnu
    return flambda


def mag_to_flambda(mag: np.ndarray, wave: np.ndarray) -> np.ndarray:
    """Convert AB magnitudes to F_lambda in erg/s/AA/cm^2.
    In the AB magnitude system, an object with flux density F_nu = 3631 Jansky
    has a magnitude of zero in all bands (note: 1 Jansky = 1e-23 erg/s/Hz/cm^2).
    Thus, to convert from AB magnitude to F_nu, we do the following:
    F_nu = (3631e-23 erg/s/Hz/cm^2) * 10^(m_AB / -2.5).
    To convert from F_nu to F_lambda, we use the formula
    F_lambda = c / lambda^2 * F_nu,
    resulting in an SED with units of erg/s/cm^2/AA.
    """
    fnu = mag_to_fnu(mag)
    flambda = fnu_to_flambda(fnu, wave)
    return flambda


def flambda_to_fnu(flambda: np.ndarray, wave: np.ndarray) -> np.ndarray:
    """Convert F_lambda in erg/s/AA/cm^2 to F_nu in erg/s/Hz/cm^2.
    To convert from F_lambda to F_nu, we use the formula
    F_nu = lambda^2 / c * F_lambda,
    resulting in an SED with units of erg/s/Hz/cm^2.
    """
    fnu = wave ** 2 / 2.998e18 * flambda
    return fnu


def fnu_to_mag(fnu: np.ndarray) -> np.ndarray:
    """Convert F_nu in erg/s/Hz/cm^2 to AB magnitudes.
    In the AB magnitude system, an object with flux density F_nu = 3631 Jansky
    has a magnitude of zero in all bands (note: 1 Jansky = 1e-23 erg/s/Hz/cm^2).
    Thus, to convert from F_nu to AB magnitude, we do the following:
    m_AB = -2.5 * log10(F_nu / 3631e-23 erg/s/Hz/cm^2).
    """
    mag = -2.5 * jnp.log10(fnu / 3631e-23)
    return mag


def flambda_to_mag(flambda: np.ndarray, wave: np.ndarray) -> np.ndarray:
    """Convert F_lambda in erg/s/AA/cm^2 to AB magnitudes.
    To convert from F_lambda to F_nu, we use the formula
    F_nu = lambda^2 / c * F_lambda,
    resulting in an SED with units of erg/s/Hz/cm^2.
    In the AB magnitude system, an object with flux density F_nu = 3631 Jansky
    has a magnitude of zero in all bands (note: 1 Jansky = 1e-23 erg/s/Hz/cm^2).
    Thus, to convert from F_nu to AB magnitude, we do the following:
    m_AB = -2.5 * log10(F_nu / 3631e-23 erg/s/Hz/cm^2).
    """
    fnu = flambda_to_fnu(flambda, wave)
    mag = fnu_to_mag(fnu)
    return mag


def wave_to_freq(wave: np.ndarray) -> np.ndarray:
    """Convert wavelength in Angstroms (AA) to frequency in Hertz (Hz)."""
    freq = 2.998e18 / wave
    return freq


def freq_to_wave(freq: np.ndarray) -> np.ndarray:
    """Convert frequency in Hertz (Hz) to wavelength in Angstroms (AA)."""
    wave = 2.998e18 / freq
    return wave
