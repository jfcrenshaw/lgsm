import pickle

import elegy
import matplotlib.pyplot as plt
import numpy as np
from lgsm.plotting import plot_photometry, plot_sed

# get the values injected to global by snakemake
model_dir = snakemake.input[1]
data_file = snakemake.input[2]
sims_file = snakemake.input[3]
output_file = snakemake.output[0]
config = snakemake.config["plotting"]["model_predictions"]
val_split = snakemake.config["lgsm"]["training"]["validation_split"]
sed_unit = snakemake.config["lgsm"]["vae"]["sed_unit"]
bandpasses = snakemake.config["lgsm"]["physics_layer"]["bandpasses"]


# make sure ncols isn't in the subplots_settings
if "ncols" in config["subplots_settings"]:
    raise ValueError(
        "Do not put ncols in subplots_settings for model_predictions. "
        "Provide ncols_train and ncols_val instead. "
        "See the default config for an example."
    )
# if it's not, calculate its value from the training and validation ncols
else:
    config["subplots_settings"]["ncols"] = config["ncols_train"] + config["ncols_val"]

# load the data
with open(data_file, "rb") as file:
    data = pickle.load(file)["data"]

# get the training and validation sets
idx_split = int(data.shape[0] * (1 - val_split))
train_set, val_set = data[:idx_split], data[idx_split:]

# set an rng to select random galaxies for plotting
rng = np.random.default_rng(config["seed"])

# select random training galaxies for plotting
ntrain = config["subplots_settings"]["nrows"] * config["ncols_train"]
train_set = rng.choice(train_set, size=ntrain, replace=False)

# select random validation galaxies for plotting
nval = config["subplots_settings"]["nrows"] * config["ncols_val"]
val_set = rng.choice(val_set, size=nval, replace=False)

# concatenate the sets
data = np.concatenate((train_set, val_set))

# now we will interleave the sets so that training sets and validation sets
# are sorted by column instead of row
nrows = config["subplots_settings"]["nrows"]
ncols = config["subplots_settings"]["ncols"]
data = data.reshape(ncols, nrows, -1)
data = np.transpose(data, (1, 0, 2))
data = data.reshape(nrows * ncols, -1)

# pull out the columns in the data
keys = data[:, 0]
amps = data[:, 1]
redshift = data[:, 2]
photometry = data[:, 3:]

# load the trained model and make predictions
predictions = elegy.load(model_dir).predict(
    np.hstack((redshift.reshape(-1, 1), photometry))
)

# finally, let's load the truth data
with open(sims_file, "rb") as file:
    sims = pickle.load(file)
    sim_wave = sims["wave"]
    sim_mag = sims["sed_mag"]

# add the simulated mags to the SEDs
sim_mag = sim_mag[keys.astype(int)] + amps[:, None]

# create the figure
fig, axes = plt.subplots(**config["subplots_settings"])

# loop through the axes and galaxies to make the plots
for i, ax in enumerate(axes.flatten()):

    sed = predictions[f"sed_{sed_unit}"][i]
    amp = predictions["amplitude"][i]
    if sed_unit == "mag":
        sed = amp + sed
    else:
        sed = amp * sed

    # plot the true sed
    plot_sed(
        sim_wave,
        sim_mag[i],
        sed_unit=sed_unit,
        plot_unit=config["plot_unit"],
        plot_settings=config["truth"]["sed_settings"],
        ax=ax,
    )

    # plot the true photometry if plot_unit = mag
    if config["plot_unit"] == "mag":
        plot_photometry(
            photometry[i],
            bandpasses,
            z=redshift[i],
            scatter_settings=config["truth"]["photometry_settings"],
            ax=ax,
        )

    # plot the predicted sed
    plot_sed(
        predictions["sed_wave"],
        sed,
        sed_unit=sed_unit,
        plot_unit=config["plot_unit"],
        plot_settings=config["predicted"]["sed_settings"],
        ax=ax,
    )

    # plot the predicted photometry if plot_unit = mag
    if config["plot_unit"] == "mag":
        plot_photometry(
            predictions["predicted_photometry"][i],
            bandpasses,
            z=redshift[i],
            scatter_settings=config["predicted"]["photometry_settings"],
            ax=ax,
        )

    # invert the y axis if we're plotting magnitudes
    if config["plot_unit"] == "mag":
        ax.invert_yaxis()

    # need to set column names to train and validation
    if i < config["ncols_train"]:
        ax.set(title="Train")
    elif i < config["ncols_train"] + config["ncols_val"]:
        ax.set(title="Test")

    # apply axis settings
    ax.set(**config["ax_settings"])

    # remove x axis and y axis labels from interior plots
    if i % ncols != 0:
        ax.set(ylabel="")
    if i < ncols * (nrows - 1):
        ax.set(xlabel="")

fig.savefig(output_file)
