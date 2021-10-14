"""Plots the latent variables of the LGS Model."""
import pickle

import corner
import elegy
import jax.numpy as jnp
import numpy as np

# get the values injected to global by snakemake
# pylint: disable=undefined-variable
model_dir = snakemake.input[1]
training_data = snakemake.input[2]
output_file = snakemake.output[0]
config = snakemake.config["plotting"]["latent_variables"]
lgsm_config = snakemake.config["lgsm"]
# pylint: enable=undefined-variable


# load the data
with open(training_data, "rb") as file:
    sims = pickle.load(file)

    # load the simulated photometry and redshifts
    redshift = jnp.array(sims["redshift"])
    photometry = jnp.array(sims["photometry"])
    data = jnp.hstack((redshift.reshape(-1, 1), photometry))

# split the training and validation sets
val_split = lgsm_config["training"]["validation_split"]
idx_split = int(redshift.size * (1 - val_split))
train_set = data[:idx_split]
val_set = data[idx_split:]

# load the trained model
model = elegy.load(model_dir)

train_latents = np.array(model.predict(train_set)["intrinsic_latents"])
val_latents = np.array(model.predict(val_set)["intrinsic_latents"])

fig = corner.corner(train_latents, color="C0", hist_kwargs={"density": True})
corner.corner(val_latents, fig=fig, color="C1", hist_kwargs={"density": True})

fig.savefig(output_file)
