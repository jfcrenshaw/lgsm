"""
Calculates simulated photometry for the provided SEDs at a variety of
redshifts and magnitudes.
"""
import pickle

import jax.numpy as jnp
from jax import random
from lgsm import PhysicsLayer


# get values injected to global by snakemake
input_file = snakemake.input[0]
output_file = snakemake.output[0]
config = snakemake.config["sims"]


# load the simulated SEDs
with open(input_file, "rb") as file:
    simulated_seds = pickle.load(file)

# get the wavelength grid and the mags for the SEDs
wave = simulated_seds["wave"]
mags = simulated_seds["sed_mag"]

# how many duplicates of each should we train?
N = config["redshifts_per_sed"]
# these keys allow us to map simulated photometry back to the true SED
keys = jnp.repeat(jnp.arange(mags.shape[0]), N).reshape(-1, 1)
# make duplicates of each SED
mags = jnp.tile(mags, N).reshape(-1, mags.shape[1])

# generate random redshifts and amplitudes for the SEDs
PRNGKey = random.PRNGKey(config["random_seed"])
zs_key, amps_key = random.split(PRNGKey)
zs = random.uniform(
    key=zs_key,
    shape=(mags.shape[0], 1),
    minval=config["min_redshift"],
    maxval=config["max_redshift"],
)
amps = random.uniform(
    key=amps_key,
    shape=(mags.shape[0], 1),
    minval=config["min_mag"],
    maxval=config["max_mag"],
)

# build the physics layer
PL = PhysicsLayer(
    sed_min=wave.min(),
    sed_max=wave.max(),
    sed_bins=wave.size,
    sed_unit="mag",
    bandpasses=config["bandpasses"],
    band_oversampling=3,
)

# simulate the photometry
photometry = PL.call(mags, amps, zs)["predicted_photometry"]

# stack all the data to save
data = jnp.hstack((keys, amps, zs, photometry))

# randomly shuffle the data
data = random.permutation(PRNGKey, data)

# save the simulations in a dictionary
bandpasses_str = ", ".join(config["bandpasses"])
save_dict = {
    "README": (
        "Simulated photometry for the SEDs in data/raw/simulated_seds.pkl. "
        f"The columns are key, amplitude, redshift, {bandpasses_str}. "
        "The key tells you which SED in simulated_seds.pkl was used to "
        "generate the photometry, while amps/zs are the amplitudes/redshifts "
        "they were simulated at."
    ),
    "data": data,
}

with open(output_file, "wb") as file:
    pickle.dump(save_dict, file)
