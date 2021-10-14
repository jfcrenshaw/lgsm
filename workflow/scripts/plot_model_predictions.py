"""Plots SED and photometry predictions for the trained LGS Model."""
import pickle

import elegy
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap
from lgsm.plotting import plot_photometry, plot_sed

# pylint: disable=undefined-variable
# get the values injected to global by snakemake
model_dir = snakemake.input[2]
training_data = snakemake.input[3]
output_file = snakemake.output[0]
config = snakemake.config["plotting"]["model_predictions"]
lgsm_config = snakemake.config["lgsm"]
# set the rcParams
plt.rcParams.update(snakemake.config["plotting"]["rcParams"])
# pylint: enable=undefined-variable


# make sure ncols isn't in the subplots_settings
if "ncols" in config["subplots_settings"]:
    raise ValueError(
        "Do not put ncols in subplots_settings for model_predictions. "
        "Provide ncols_train and ncols_val instead. "
        "See the default config for an example."
    )

# calculate ncols from the training and validation ncols
nrows = config["subplots_settings"]["nrows"]
ncols_train = config["ncols_train"]
ncols_val = config["ncols_val"]
ncols = ncols_train + ncols_val
config["subplots_settings"]["ncols"] = ncols

# load the data
with open(training_data, "rb") as file:
    sims = pickle.load(file)

    # load the simulated photometry and redshifts
    redshift = jnp.array(sims["redshift"])
    photometry = jnp.array(sims["photometry"])

    # load the true SEDs
    true_wave = jnp.array(sims["sed_wave"])
    true_seds = jnp.array(sims["sed_mag"])

# get the split of the training and validation sets
val_split = lgsm_config["training"]["validation_split"]
idx_split = int(redshift.size * (1 - val_split))

# select random sets of the training and validation sets to plot
PRNGKey = random.PRNGKey(config["galaxy_seed"])
train_key, val_key = random.split(PRNGKey)

ntrain = nrows * ncols_train
train_idx = random.choice(
    train_key, jnp.arange(0, idx_split), shape=(ntrain,), replace=False
)
# train_idx = random.randint(train_key, shape=(ntrain,), minval=0, maxval=idx_split)

nval = nrows * ncols_val
val_idx = random.choice(
    val_key, jnp.arange(idx_split, redshift.size), shape=(nval,), replace=False
)
# val_idx = random.randint(val_key, shape=(nval,), minval=idx_split, maxval=redshift.size)

# concatenate the sets
idx = jnp.concatenate((train_idx, val_idx))

# now we will order the indices so that the training and validation sets
# will be sorted by column instead of row
idx = idx.reshape(ncols, nrows).T.flatten()

# pull out the values we will plot
redshift = redshift[idx]
photometry = photometry[idx]
true_seds = true_seds[idx]

# get the list of bandpasses
bandpasses = lgsm_config["physics_layer"]["bandpasses"]

# load the trained model
model = elegy.load(model_dir)

# create the figure
fig, axes = plt.subplots(**config["subplots_settings"])

# sample from the model with different seeds, and make the plots
PRNGKey = random.PRNGKey(config["encoder_seed"])
seeds = random.split(PRNGKey, num=config["nsamples"])
for seed in seeds:

    # set the seed for the model
    model.states = model.states.update(rng=elegy.RNGSeq(seed))
    # get the new predictions
    predictions = model.predict(jnp.hstack((redshift.reshape(-1, 1), photometry)))

    # loop through the axes and galaxies to make the plots
    for i, ax in enumerate(axes.flatten()):

        sed_unit = lgsm_config["vae"]["sed_unit"]
        sed = predictions[f"sed_{sed_unit}"][i]
        amp = predictions["amplitude"][i]
        if sed_unit == "mag":
            sed = amp + sed
        else:
            sed = amp * sed

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
                redshift=redshift[i],
                scatter_settings=config["predicted"]["photometry_settings"],
                ax=ax,
            )

# downsample the true SEDs
true_seds = vmap(lambda mags: jnp.interp(predictions["sed_wave"], true_wave, mags))(
    true_seds
)
# plot the true values and set axis settings
for i, ax in enumerate(axes.flatten()):

    # plot the true sed
    plot_sed(
        predictions["sed_wave"],
        true_seds[i],
        sed_unit="mag",
        plot_unit=config["plot_unit"],
        plot_settings=config["truth"]["sed_settings"],
        ax=ax,
    )

    # plot the true photometry if plot_unit = mag
    if config["plot_unit"] == "mag":
        plot_photometry(
            photometry[i],
            bandpasses,
            redshift=redshift[i],
            scatter_settings=config["truth"]["photometry_settings"],
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
