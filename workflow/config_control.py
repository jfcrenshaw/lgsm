"""Tools to help handle snakemake configs."""
from functools import reduce
from pathlib import Path
from shutil import rmtree
from typing import Union

import yaml


class ConfigFlags:
    """Controls flagging the config file so that if sections of the config
    are changed, the corresponding rules will be re-run.
    """

    _flag_path = ".config_flags"  # path where the flags will be stored

    def __init__(
        self,
        new_config: dict,
        old_config_path: str = None,
        default_config_path: str = None,
    ):
        """
        Parameters
        ----------
        new_config : dict
            The dictionary of new config settings.
        old_config_path : str, optional
            The path to the old config file.
        default_config_path : str, optional
            The path to the default config file against which to check the
            new config, if there is no config at old_config_path.
        """

        # save the new config
        self._new_config = new_config

        # set the reference config
        # if the old config exists, use that as the reference config
        if old_config_path is not None and Path(old_config_path).is_file():
            with open(old_config_path, "r") as file:
                self._ref_config = yaml.safe_load(file)
        # else if a default config is passed, used that as the ref config
        elif default_config_path is not None and Path(default_config_path).is_file():
            with open(default_config_path, "r") as file:
                self._ref_config = yaml.safe_load(file)
        # else, there is no reference
        else:
            raise ValueError(
                "You must provide the path to an old config or a default "
                "config against which to compare the new config."
            )

        # create directory to hold the flags
        Path(self._flag_path).mkdir(exist_ok=True)

        # set all_lowered to False
        self._all_lowered = False

    @staticmethod
    def _get_nested_value(dictionary, keys):
        """Get value from nested dictionary keys."""
        return reduce(lambda d, k: d[k], keys, dictionary)

    def flag(self, *keys: str) -> Union[str, list]:
        """Set a flag for the given config key(s) in a snakemake rule input."""

        # get the new and reference config values for this chain of keys
        new_val = self._get_nested_value(self._new_config, keys)
        ref_val = self._get_nested_value(self._ref_config, keys)

        # compare the config values...
        # if they are the same, or self.lower_all() has been called,
        # do not raise a flag
        if new_val == ref_val or self._all_lowered:
            return []
        # else raise a flag
        else:
            flag_file = f"{self._flag_path}/{'_'.join(keys)}_changed"
            Path(flag_file).touch()
            return flag_file

    def lower_all(self):
        """Lower all the flags regardless of config changes."""
        self._all_lowered = True

    def cleanup(self):
        """Remove all the flag files."""
        rmtree(self._flag_path)


def _return_ordered_config(unordered_config: dict, template_config: dict) -> dict:
    """Orders the keys in unordered_config according to the order in template_config.

    Note this function assumes all keys in both dictionaries are identical,
    including all nested dictionaries.
    """

    # create a dictionary for the ordered config
    ordered_config = {}

    # loop through keys in the order of the template config
    for key in template_config.keys():

        # get the value from the unordered config
        val = unordered_config[key]

        # if the value is a dictionary, it needs to be ordered too
        if isinstance(val, dict):
            ordered_config[key] = _return_ordered_config(val, template_config[key])
        # otherwise just save the value
        else:
            ordered_config[key] = val

    return ordered_config


def save_config(config: dict, output_path: str, template_path: str = None):
    """Saves the config at the designated path.

    Parameters
    ----------
    config : dict
        The config dictionary to save
    output_path : str
        The path where the config is saved
    template_path : str, optional
        If provided, the config is saved with keys saved in the same order
        as the template config (including all nested dictionaries)
    """

    # if template is None, call return_ordered_config to make sure dict doesn't
    # contain an OrderedDict
    if template_path is None:
        config = _return_ordered_config(config, config)
    # but if a template is provided, we order the config according to the template
    else:
        with open(template_path, "r") as file:
            template_config = yaml.safe_load(file)
        config = _return_ordered_config(config, template_config)

    # save the config
    with open(output_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)
