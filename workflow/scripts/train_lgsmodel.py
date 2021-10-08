"""Trains an LGS Model."""
import pickle

import elegy
import jax.numpy as jnp
import optax
from lgsm import LGSModel, losses

# get the values injected to global by snakemake
# pylint: disable=undefined-variable
input_file = snakemake.input[1]
output_dir = snakemake.output[0]
config = snakemake.config["lgsm"]
# pylint: enable=undefined-variable

# load the data
with open(input_file, "rb") as file:
    data = pickle.load(file)["data"]
    data = jnp.array(data[:, 2:])

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
    y=jnp.ones(data.shape),
    sample_weight=jnp.ones(data.shape),
    **config["training"],
)

# save the model
model.save(output_dir)
