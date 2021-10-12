"""Trains an LGS Model."""
import pickle

import elegy
import jax
import jax.numpy as jnp
import optax
from lgsm import LGSModel, losses, sed_utils

# get the values injected to global by snakemake
# pylint: disable=undefined-variable
training_data_file = snakemake.input[1]
sims_file = snakemake.input[2]
model_dir = snakemake.output[0]
config = snakemake.config["lgsm"]
# pylint: enable=undefined-variable

# load the data
with open(training_data_file, "rb") as file:
    data = pickle.load(file)["data"]
    keys = jnp.array(data[:, 0]).astype(int)
    amps = jnp.array(data[:, 1])[:, None]
    data = jnp.array(data[:, 2:])

# if we are using SpectralLoss, we need to load the simulations
if config["training"]["losses"]["SpectralLoss"]["use"]:

    # load the sims, in case we are using SpectralLoss
    with open(sims_file, "rb") as file:
        sims = pickle.load(file)
        sim_wave = sims["wave"]
        sim_mag = sims["sed_mag"]

    # get the relevant SEDs and add the simulated amplitudes
    sim_mag = sim_mag[keys] + amps

    # down-sample the simulations to the resolution of the predicted latent SEDs
    sed_wave = sed_utils.setup_wave_grid(
        config["vae"]["wave_min"], config["vae"]["wave_max"], config["vae"]["wave_bins"]
    )
    y = jax.vmap(lambda mags: jnp.interp(sed_wave, sim_wave, mags))(sim_mag)

# otherwise, create a fake y
else:
    y = jnp.ones(data.shape[0])

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
        data.mean(axis=0),
        data.std(axis=0),
        **config["vae"],
        **config["physics_layer"],
    ),
    loss=losses,
    optimizer=optimizer,
)

# train the model
# currently y and sample weight are dummies to allow validation_split to work
history = model.fit(
    x=data,
    verbose=2,
    shuffle=False,
    y=y,
    sample_weight=jnp.ones(data.shape),
    **config["training"],
)

# save the model
model.save(model_dir)
