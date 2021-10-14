"""Trains an LGS Model."""
import pickle

import elegy
import jax
import jax.numpy as jnp
import optax
from lgsm import LGSModel, losses, sed_utils

# get the values injected to global by snakemake
# pylint: disable=undefined-variable
training_data = snakemake.input[1]
model_dir = snakemake.output[0]
config = snakemake.config["lgsm"]
# pylint: enable=undefined-variable

# load the data
with open(training_data, "rb") as file:
    sims = pickle.load(file)
    redshift = sims["redshift"]
    photometry = sims["photometry"]
    training_data = jnp.hstack((redshift, photometry))

# if we are using SpectralLoss, we need to load the true SEDs associated
# with the photometry simulations
if config["training"]["losses"]["SpectralLoss"]["use"]:

    true_seds = sims["sed_mag"]
    true_wave = sims["sed_wave"]

    # down-sample the true SEDs to the resolution of the predicted latent SEDs
    sed_wave = sed_utils.setup_wave_grid(
        config["vae"]["wave_min"], config["vae"]["wave_max"], config["vae"]["wave_bins"]
    )
    y = jax.vmap(lambda mags: jnp.interp(sed_wave, true_wave, mags))(true_seds)

# otherwise, create a fake y
else:
    y = jnp.ones(training_data.shape[0])

# build the list of loss functions
losses = [
    getattr(losses, loss_name)(**loss_settings["params"])
    for loss_name, loss_settings in config["training"].pop("losses").items()
    if loss_settings["use"]
]

# get the optimizer
optimizer_name, optimizer_params = config["training"].pop("optimizer").popitem()
optimizer = getattr(optax, optimizer_name)(**optimizer_params)

# build the elegy model
model = elegy.Model(
    module=LGSModel(
        training_data.mean(axis=0),
        training_data.std(axis=0),
        **config["vae"],
        **config["physics_layer"],
    ),
    loss=losses,
    optimizer=optimizer,
)

# train the model
# sample weight is a dummy to allow validation_split to work
history = model.fit(
    x=training_data,
    y=y,
    sample_weight=jnp.ones(training_data.shape),
    verbose=2,
    shuffle=False,
    **config["training"],
)

# save the model
model.save(model_dir)
