"""Save the full config file in the results file."""
import yaml

# get values injected to global by snakemake
output_file = snakemake.output[0]
config = snakemake.config

# save the config file in the results directory
with open(output_file, "w") as file:
    yaml.dump(config, file, default_flow_style=False)
