# LGSM: Latent Galaxy SED Model

![build](https://github.com/jfcrenshaw/lgsm/workflows/build/badge.svg)

Building a physics-informed VAE model to deconvolve SED's from photometry.

This project was started in another repo.
I ported that work over here, where I am now handling dependencies with poetry, and controlling the workflow with snakemake.

To install, clone this repo, and from the root directory, run `poetry install`.
If you want to enable GPU support for Jax with CUDA, run

```bash
poetry shell
pip install --upgrade pip
pip install --upgrade jaxlib==0.1.71+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Note that the final command may need to be changed, depending on your version of Cuda (see more under the installation instructions [here](https://github.com/google/jax)).

For GPU support, you may also need to add the following to your `.bashrc`:

```bash
# cuda setup
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=$PATH:/usr/local/cuda/bin
```

If you have the GPU enabled version of jax installed, but would like to run on a CPU, add the following to the top of your scripts/notebooks:

```python
import jax
# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')
```

Note that if you run jax on GPU in multiple Jupyter notebooks simultaneously, you may get `RuntimeError: cuSolver internal error`. Read more [here](https://github.com/google/jax/issues/4497) and [here](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html).
