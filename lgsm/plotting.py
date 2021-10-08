"""Methods for plotting photometry and SEDs."""
from numbers import Number
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sncosmo

from lgsm import sed_utils


def plot_photometry(
    photometry: np.ndarray,
    bandpasses: Sequence[str],
    redshift: np.ndarray = np.zeros(1),
    fig_settings: dict = None,
    ax_settings: dict = None,
    scatter_settings: dict = None,
    ax: plt.Axes = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots the given photometry

    Parameters
    ----------
    photometry: np.ndarray
        Array of photometry to plot. If 2 dimensional, the second dimension
        must match the length of bandpasses
    bandpasses: Sequence[str]
        The names of the bandpasses for which the photometry was calculated
    redshift: np.ndarray, default=np.zeros(1)
        The redshift at which the photometry was observed. Used to shift
        photometry back to the restframe. Must contain one redshift or one
        redshift per per row in photometry.
    fig_settings: dict, optional
        Settings to pass to the subplots constructor
    ax_settings: dict, optional
        Settings to pass to ax.set()
    scatter_settings: dict, optional
        Settings to pass to ax.scatter()
    ax: plt.Axes, optional
        A matplotlib.pyplot Axes object to plot the photometry on. If not
        provided, new figure and axes are created.

    Returns
    -------
    plt.Figure
        A matplotlib.pyplot figure. If an ax was passed, this is None.
    plt.Axes
        The matplotlib.pyplot axes the photometry is plotted on.
    """

    # check that the redshift is valid
    redshift = np.array(redshift)
    if np.any(redshift < 0):
        raise ValueError("redshift must be non-negative")

    # assign empty dicts to settings that are None
    for settings in [fig_settings, ax_settings, scatter_settings]:
        if settings is None:
            settings = {}

    # if ax is passed, don't pass fig_settings
    if ax is not None and len(fig_settings) > 0:
        raise ValueError("If ax passed, don't pass fig_settings.")

    # make sure arrays are correct shape
    photometry = photometry.reshape(-1, len(bandpasses))
    redshift = redshift.reshape(-1, 1)

    # get the effective wavelengths of the bandpasses
    wave = np.array(
        photometry.shape[0]
        * [[sncosmo.get_bandpass(band).wave_eff for band in bandpasses]]
    ) / (1 + redshift)

    # shift photometry to restframe
    photometry = photometry + 2.5 * np.log10(1 + redshift)

    # if ax not provided, we will make a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(**fig_settings)
        ax.invert_yaxis()
    else:
        fig = None

    # plot the photometry
    ax.scatter(wave.flatten(), photometry.flatten(), **scatter_settings)

    # set the axis labels
    ax.set(xlabel="Wavelength (AA)", ylabel="AB Magnitude")

    # set user-supplied settings
    ax.set(**ax_settings)

    # if we made a new figure and axis, return them
    return fig, ax


def _plot_sed(
    wave: np.ndarray,
    sed: np.ndarray,
    redshift: float = 0,
    normalize_at: float = None,
    sed_unit: str = "mag",
    plot_unit: str = None,
    fig_settings: dict = None,
    ax_settings: dict = None,
    plot_settings: dict = None,
    ax: plt.Axes = None,
) -> Tuple[plt.Figure, plt.Axes]:

    # check that the redshift is valid
    if redshift < 0:
        raise ValueError("redshift must be non-negative.")
    # check that normalize at is valid
    if normalize_at is not None:
        if not isinstance(normalize_at, Number):
            raise ValueError("normalize_at must be None or a number.")
        if normalize_at < wave.min() or normalize_at > wave.max():
            raise ValueError("normalize_at must be within range of wave.")
    # check that sed_unit and plot_unit are valid
    units_allowed = ["mag", "flambda", "fnu"]
    if sed_unit not in units_allowed[:-1]:
        raise ValueError(f"sed_unit must be one of {', '.join(units_allowed[-1])}")
    if plot_unit is None:
        plot_unit = sed_unit
    if plot_unit not in units_allowed:
        raise ValueError(f"plot_unit must be one of {', '.join(units_allowed)}")

    # assign empty dicts to settings that are None
    for settings in [fig_settings, ax_settings, plot_settings]:
        if settings is None:
            settings = {}

    # redshift the sed
    wave = wave * (1 + redshift)
    if sed_unit == "mag":
        sed = sed - 2.5 * np.log10(1 + redshift)
    else:
        sed = sed / (1 + redshift)

    # (possibly) convert sed units to the plotting units
    if sed_unit != plot_unit:
        sed = getattr(sed_unit, f"{sed_unit}_to_{plot_unit}")(sed, wave=wave)
    # (possibly) convert wavelength to frequency
    if plot_unit == "fnu":
        # note if this is called, "wave" is actually an array of frequencies
        wave = sed_utils.wave_to_freq(wave)[::-1]

    # (possibly) normalize the SED at the requested location
    if normalize_at is not None:
        idx = np.argmin(np.abs(wave - normalize_at))
        if plot_unit == "mag":
            sed -= sed[idx]
        else:
            sed /= sed[idx]

    # if ax not provided, we will make a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(**fig_settings)
        # if we are plotting mags, invert the y-axis
        if plot_unit == "mag":
            ax.invert_yaxis()
    else:
        fig = None

    # plot the sed
    ax.plot(wave, sed, **plot_settings)

    # set the axis labels depending on plot_unit
    if plot_unit == "mag":
        xlabel = "Wavelength (AA)"
        ylabel = "AB Magnitude"
    elif plot_unit == "flambda":
        xlabel = "Wavelength (AA)"
        ylabel = "Flux Density (erg/s/AA/cm^2)"
    else:
        xlabel = "Frequency (Hz)"
        ylabel = "Flux Density (erg/s/Hz/cm^2)"
    ax.set(xlabel=xlabel, ylabel=ylabel)

    # set the user-supplied axis settings
    ax.set(**ax_settings)

    return fig, ax


def plot_sed(
    wave: np.ndarray,
    sed: np.ndarray,
    redshift: np.ndarray = np.zeros(1),
    normalize_at: float = None,
    sed_unit: str = "mag",
    plot_unit: str = None,
    fig_settings: dict = None,
    ax_settings: dict = None,
    plot_settings: dict = None,
    ax: plt.Axes = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots the given seds

    Attempts to broadcast all of the inputs so you can plot many SEDs at once.

    Parameters
    ----------
    wave: np.ndarray
        The wavelength grid(s)
    sed: np.ndarray
        The SED(s)
    redshift: np.ndarray, default=np.zeros(1)
        The redshift at which to plot the SED.
    normalize_at: float, optional
        The wavelength at which to normalize the SEDs
    sed_unit: str, default="mag"
        The unit of the input seds. Can be "mag" or "flambda"
    plot_unit: str, optional
        The unit in which to plot the seds. Can be "mag", "flambda", or
        "fnu". If None, assumed to be same as sed_unit
    fig_settings: dict, optional
        Settings to pass to the subplots constructor
    ax_settings: dict, optional
        Settings to pass to ax.set()
    plot_settings: dict, optional
        Settings to pass to ax.scatter(). If doesn't exactly match number
        of SEDs to plot, these settings will define property cycles.
    ax: plt.Axes, optional
        A matplotlib.pyplot Axes object to plot the photometry on. If not
        provided, new figure and axes are created.

    Returns
    -------
    plt.Figure
        A matplotlib.pyplot figure. If an ax was passed, this is None.
    plt.Axes
        The matplotlib.pyplot axes the photometry is plotted on.
    """

    # assign empty dicts to settings that are None
    for settings in [fig_settings, ax_settings, plot_settings]:
        if settings is None:
            settings = {}

    # if ax is passed, don't pass fig_settings
    if ax is not None and len(fig_settings) > 0:
        raise ValueError("If ax passed, don't pass fig_settings.")

    # save the settings that all function calls receive
    settings = {
        "normalize_at": normalize_at,
        "sed_unit": sed_unit,
        "plot_unit": plot_unit,
        "fig_settings": fig_settings,
        "ax_settings": ax_settings,
    }

    # enforce correct shapes
    wave = np.atleast_2d(wave)
    sed = np.atleast_2d(sed)
    redshift = np.reshape(redshift, (-1, 1))

    # if multiple wavelength grid, seds, or redshifts we provided
    # then we need to plot in a loop over them
    if wave.shape[0] > 1 or sed.shape[0] > 1 or redshift.size > 1:

        # broadcast wave and sed to a common shape
        wave, sed = np.broadcast_arrays(wave, sed)

        # if only one redshift, broadcast to shape for wave, sed
        if redshift.size == 1:
            redshift = np.broadcast_to(redshift, (wave.shape[0], 1))
        # if multiple redshifts, plot each sed at each redshift
        else:
            wave, sed, redshift = (
                np.tile(wave, redshift.size).reshape(-1, wave.shape[1]),
                np.tile(sed, redshift.size).reshape(-1, sed.shape[1]),
                np.tile(redshift.flatten(), wave.shape[0]).reshape(-1, 1),
            )

        # broadcast plot_settings too
        for key, vals in plot_settings.items():
            vals = np.array([vals]).flatten()
            vals = np.tile(vals, np.ceil(redshift.size / vals.size).astype("int"))
            plot_settings[key] = vals

        # if no ax supplied, the first call will create an ax, which we
        # will pass to all subsequent calls. We will also return the new
        # fig and ax
        if ax is None:
            fig, ax = _plot_sed(
                wave=wave[0],
                sed=sed[0],
                redshift=redshift[0],
                plot_settings={
                    key: plot_settings[key][0] for key in plot_settings.keys()
                },
                **settings,
            )
            for i, (W, S, Z) in enumerate(zip(wave[1:], sed[1:], redshift[1:])):
                _plot_sed(
                    wave=W,
                    sed=S,
                    redshift=Z,
                    ax=ax,
                    plot_settings={
                        key: plot_settings[key][i + 1] for key in plot_settings.keys()
                    },
                    **settings,
                )
            return fig, ax
        # if an ax was supplied, we can do a simple loop and provide it every time
        else:
            for i, (W, S, Z) in enumerate(zip(wave, sed, redshift)):
                fig, ax = _plot_sed(
                    wave=W,
                    sed=S,
                    redshift=Z,
                    ax=ax,
                    plot_settings={
                        key: plot_settings[key][i] for key in plot_settings.keys()
                    },
                    **settings,
                )
            return fig, ax

    # otherwise, just plot the single set given
    else:

        return _plot_sed(
            wave=wave[0],
            sed=sed[0],
            redshift=redshift[0],
            ax=ax,
            plot_settings=plot_settings,
            **settings,
        )
