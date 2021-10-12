"""
Calculates simulated photometry for the provided SEDs at a variety of
redshifts and magnitudes.
"""
import pickle

import jax.numpy as jnp
from jax import random
from lgsm import PhysicsLayer

# get values injected to global by snakemake
# pylint: disable=undefined-variable
input_file = snakemake.input[1]
output_file = snakemake.output[0]
config = snakemake.config["sims"]
# pylint: enable=undefined-variable


# load the simulated SEDs
with open(input_file, "rb") as file:
    simulated_seds = pickle.load(file)

# get the wavelength grid and the mags for the SEDs
wave = simulated_seds["wave"]
mags = simulated_seds["sed_mag"]

# how many duplicates of each should we train?
N = config["redshifts_per_sed"]
# make duplicates of each SED
mags = jnp.tile(mags, N).reshape(-1, mags.shape[1])

# generate the random keys used below
PRNGKey = random.PRNGKey(config["random_seed"])
permutation_key, redshift_key, amplitude_key = random.split(PRNGKey, num=3)

# shuffle the SEDs
mags = random.permutation(permutation_key, mags)
# generate random redshifts
redshifts = zs = random.uniform(
    key=redshift_key,
    shape=(mags.shape[0], 1),
    minval=config["min_redshift"],
    maxval=config["max_redshift"],
)
# generate random amplitudes
amplitudes = random.uniform(
    key=amplitude_key,
    shape=(mags.shape[0], 1),
    minval=config["min_mag"],
    maxval=config["max_mag"],
)

# build the physics layer
PL = PhysicsLayer(
    sed_wave=wave,
    sed_unit="mag",
    bandpasses=config["bandpasses"],
    band_oversampling=3,
)

# simulate the photometry
photometry = PL.call(mags, amplitudes, redshifts)["predicted_photometry"]

# save the simulations in a dictionary
bandpasses_str = ", ".join(config["bandpasses"])
save_dict = {
    "README": (
        f"Simulated photometry for the SEDs in {input_file}. "
        "'sed_mag' is an array of the true SEDs in AB mag. "
        "'sed_wave' is an array of the corresponding wavelength grid in Angstroms. "
        "'photometry' is the photometry in AB mags, "
        f"calculated for the bands {bandpasses_str}. "
        "The photometry was calculated at the redshifts stored in 'redshift'."
    ),
    "sed_wave": wave,
    "sed_mag": mags + amplitudes,
    "redshift": redshifts,
    "photometry": photometry,
}

with open(output_file, "wb") as file:
    pickle.dump(save_dict, file)
