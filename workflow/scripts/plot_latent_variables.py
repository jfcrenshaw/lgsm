"""Plots the latent variables of the LGS Model."""
import pickle

import elegy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lgsm import PhysicsLayer
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# pylint: disable=undefined-variable
# get the values injected to global by snakemake
model_dir = snakemake.input[2]
training_data = snakemake.input[3]
output_file = snakemake.output[0]
config = snakemake.config["plotting"]["latent_variables"]
lgsm_config = snakemake.config["lgsm"]
# set the rcParams
plt.rcParams.update(snakemake.config["plotting"]["rcParams"])
# pylint: enable=undefined-variable


# get the list of bandpasses
bandpasses = lgsm_config["physics_layer"]["bandpasses"]


# load the data
with open(training_data, "rb") as file:
    sims = pickle.load(file)

    # load the simulated photometry and redshifts
    data = pd.DataFrame(
        np.hstack((sims["redshift"].reshape(-1, 1), sims["photometry"])),
        columns=["redshift"] + bandpasses,
    )

    # load the simulated SEDs
    sim_wave = sims["sed_wave"]
    sim_sed = sims["sed_mag"]

# mark the training and validation sets
val_split = 0.2
ntrain = int(data.shape[0] * (1 - val_split)) + 1
data["set"] = ntrain * ["training"] + (data.shape[0] - ntrain) * ["validation"]

# mark which galaxies had their spectra used during training
batch_size = lgsm_config["training"]["batch_size"]
spectrum_used = data.shape[0] * [False]
spectrum_used[: ntrain - 1 : batch_size] = [True] * len(
    spectrum_used[: ntrain - 1 : batch_size]
)
data["spectrum"] = spectrum_used

# set the number of points to plot
N = config["npoints"]

# now we will select a subset from data
data = pd.concat(
    (
        data.query("set == 'training' & spectrum == True")[: N // 2],
        data.query("set == 'training' & spectrum == False")[: N // 2],
        data.query("set == 'validation'")[: N // 2],
    )
)

# now let's calculate observed colors
colors = [f"{bandpasses[i]}-{bandpasses[i+1]}" for i in range(len(bandpasses) - 1)]
for i, color in enumerate(colors):
    data[color] = data[bandpasses[i]] - data[bandpasses[i + 1]]

# let's also calculate the restframe colors
PL = PhysicsLayer(
    sed_wave=sim_wave,
    sed_unit="mag",
    bandpasses=bandpasses,
    band_oversampling=3,
)
restframe_photometry = PL.call(
    sim_sed[data.index.tolist()],
    np.zeros(data.shape[0]).reshape(-1, 1),
    np.zeros(data.shape[0]).reshape(-1, 1),
)["predicted_photometry"]

restframe_colors = [f"restframe_{color}" for color in colors]
data[restframe_colors] = np.diff(restframe_photometry[::-1], axis=-1)

# finally let's strip lsst from all the columns names for cleaner plotting
data.columns = [col.replace("lsst", "") for col in data.columns.tolist()]
bandpasses = [band.replace("lsst", "") for band in bandpasses]


# now load the trained model
model = elegy.load(model_dir)

# make predictions
predictions = model.predict(data[["redshift"] + bandpasses].to_numpy())

# save the latents
latents = [f"s{i}" for i in range(predictions["latent_mean"].shape[1])]
data[latents] = predictions["latent_mean"]

# and the MSE
data["mse"] = np.mean(
    (
        0.5
        / 0.05 ** 2
        * (predictions["predicted_photometry"] - data[bandpasses].to_numpy()) ** 2
    ),
    axis=-1,
)


# now we have calculated all the values we want to plot,
# it's time to start making plots!

sns.set_theme(style="ticks")

figures = []

# (1) plot latent variables by data set
# -------------------------------------

pg = sns.pairplot(
    data=data.query("spectrum == False"),
    hue="set",
    markers=".",
    corner=True,
    vars=["s0", "s1", "s2"],
)

handles = pg._legend_data.values()
labels = pg._legend_data.keys()
pg._legend.remove()
pg.fig.legend(handles, labels, loc=(0.7, 0.82))

pg.fig.suptitle("Latent variables, colored by training/validation set", y=1.03)

for ax in pg.axes.flatten():
    if ax is not None:
        ax.set_rasterized(True)

figures.append(pg)


# (2) plot latent variables by whether the spectrum was used in training
# ----------------------------------------------------------------------

if lgsm_config["training"]["losses"]["SpectralLoss"]["use"]:
    pg = sns.pairplot(
        data=pd.concat(
            (
                data.query("spectrum == True")[: N // 2],
                data.query("spectrum == False & set == 'training'")[: N // 4],
                data.query("spectrum == False & set == 'validation'")[: N // 4],
            )
        ),
        hue="spectrum",
        hue_order=[True, False],
        markers=".",
        corner=True,
        vars=["s0", "s1", "s2"],
    )

    handles = pg._legend_data.values()
    labels = pg._legend_data.keys()
    pg._legend.remove()
    pg.fig.legend(
        handles, labels, loc=(0.65, 0.78), title="True spectrum\n   in training"
    )

    pg.fig.suptitle(
        "Latent variables, colored by whether true spectrum was used in training",
        y=1.03,
    )

    for ax in pg.axes.flatten():
        if ax is not None:
            ax.set_rasterized(True)

    figures.append(pg)


# (2) plot latent variables colored by the list of variables below
# ----------------------------------------------------------------


def pairplot_latents(hue_var, query="(spectrum == False) & (mse < 10)"):
    cmap = "magma_r"

    pg = sns.pairplot(
        data=data.query(query),
        hue=hue_var,
        markers=".",
        palette=cmap,
        x_vars=["s0", "s1"],
        y_vars=["s1", "s2"],
        diag_kind=None,
    )

    pg._legend.remove()
    norm = plt.Normalize(
        data.query(query)[hue_var].min(), data.query(query)[hue_var].max()
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    pg.axes[0, 1].remove()

    axins = inset_axes(
        pg.axes[1, 1], width="90%", height="20%", loc="upper center", borderpad=-10
    )
    pg.fig.colorbar(sm, cax=axins, label=hue_var, orientation="horizontal")

    pg.fig.suptitle(f"Latent variables, colored by {hue_var}", y=1.03)

    for ax in pg.axes.flatten():
        ax.set_rasterized(True)
    # axins.set_rasterized(True)

    return pg


hue_vars = [
    "mse",
    "redshift",
    "u-g",
    "g-r",
    "r-i",
    "i-z",
    "z-y",
    "restframe_u-g",
    "restframe_g-r",
    "restframe_r-i",
    "restframe_i-z",
    "restframe_z-y",
]

for var in hue_vars:
    figures.append(pairplot_latents(var))


# NOW save all these figs in a single pdf
# ---------------------------------------
with PdfPages(output_file) as pdf:
    for pg in figures:
        pdf.savefig(pg.fig, bbox_inches="tight", dpi=200)
