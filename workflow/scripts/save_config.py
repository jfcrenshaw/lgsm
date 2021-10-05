"""Save the full config file in the results file."""
import yaml

# get values injected to global by snakemake
input_file = snakemake.input[0]
output_file = snakemake.output[0]
config = snakemake.config


# define a function that will order the config according to the template yaml
def return_ordered_config(template_config: dict, unordered_config: dict) -> dict:
    """Orders the unordered_config dict in the order of template_config.

    Note it assumes all the keys in both dictionaries are the same, including
    all nested dictionaries.
    """

    # create a new dictionary to contain the ordered config
    ordered_config = {}

    # loop through keys in the order of template_dict
    for key in template_config.keys():

        # get the value from the unordered_config
        val = unordered_config[key]

        if isinstance(val, dict):
            ordered_config[key] = return_ordered_config(template_config[key], val)
        else:
            ordered_config[key] = val

    return ordered_config


# load the template yaml
with open(input_file, "r") as file:
    default_config = yaml.safe_load(file)

# order the config in the order specified in the template yaml
config = return_ordered_config(default_config, config)

# save the config file in the results directory
with open(output_file, "w") as file:
    yaml.dump(config, file, default_flow_style=False, sort_keys=False)
