"""Plots the training losses of the LGS Model."""
import elegy
import matplotlib.pyplot as plt

# pylint: disable=undefined-variable
# get the values injected to global by snakemake
input_dir = snakemake.input[2]
output_file = snakemake.output[0]
config = snakemake.config["plotting"]["model_losses"]
# set the rcParams
plt.rcParams.update(snakemake.config["plotting"]["rcParams"])
# pylint: enable=undefined-variable

# load the training history of the elegy model
history = elegy.load(input_dir).history.history

# determine how many plots we need
nplots = len(history.keys()) // 2
subplot_settings = config["subplots_settings"]
if "nrows" not in subplot_settings or "ncols" not in subplot_settings:
    subplot_settings["nrows"] = nplots
    subplot_settings["ncols"] = 1
assert (
    subplot_settings["nrows"] * subplot_settings["ncols"] >= nplots
), "Not enough rows and columns to plot the losses."

# if figsize not provided, set default
if "figsize" not in subplot_settings:
    subplot_settings["figsize"] = [4, 2 * nplots]

# set up the subplots
fig, axes = plt.subplots(**subplot_settings)

# plot the losses
for i in range(nplots):

    # get the axis
    ax = axes.flatten()[i]

    # get the key
    key = list(history.keys())[i]

    # plot the training metric
    metric = history[key]
    ax.plot(metric, label=f"Training {key}")

    # plot the validation metric
    val_metric = history[f"val_{key}"]
    ax.plot(val_metric, label=f"Validation {key}")

    # plot settings
    ax.set(
        ylabel=key, title=f"Training and Validation {key}", **config["plot_settings"]
    )
    ax.legend()

# save the figure
fig.savefig(output_file)
