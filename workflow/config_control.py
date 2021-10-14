"""Tools to help handle snakemake configs."""
from functools import reduce
from pathlib import Path
from shutil import rmtree
from typing import Union

import yaml
from snakemake.io import ancient, AnnotatedString


class ConfigFlags:
    """Controls flagging the config file so that if sections of the config
    are changed, the corresponding rules will be re-run.
    """

    _flag_path = ".config_flags"  # path where the flags will be stored

    def __init__(self, new_config: dict, old_config_path: str):
        """
        Parameters
        ----------
        new_config : dict
            The dictionary of new config settings.
        old_config_path : str, optional
            The path to the old config file.
        """

        # set all_lowered and all_raised to False
        self._all_lowered = False
        self._all_raised = False

        # save the new config
        self._new_config = new_config

        # set the reference config
        if Path(old_config_path).is_file():
            with open(old_config_path, "r") as file:
                self._ref_config = yaml.safe_load(file)
        else:
            self._ref_config = None

        # create directory to hold the flags
        Path(self._flag_path).mkdir(exist_ok=True)

    @staticmethod
    def _get_nested_value(dictionary, keys):
        """Get value from nested dictionary keys."""
        return reduce(lambda d, k: d[k], keys, dictionary)

    def _raised_flag(self, keys: tuple) -> str:
        """Returns the name of a raised flag built from the given keys."""
        return f"{self._flag_path}/{'_'.join(keys)}_changed"

    def _lowered_flag(self) -> Union[AnnotatedString, list]:
        """Returns the name of the lowered flag.

        The lowered flag is ancient so that snakemake considers the it older
        than all outputs, and thus the rule is not triggered.
        """
        return ancient(f"{self._flag_path}/_lowered_flag")

    def flag(self, *keys: str) -> Union[str, list]:
        """Set a flag for the given config key(s) in a snakemake rule input."""

        # if all_raised() has been called or there is no ref_config, raise the flag
        if self._all_raised or self._ref_config is None:
            flag_file = self._raised_flag(keys)

        # else if all_lowered() has been called, lower the flag
        elif self._all_lowered:
            flag_file = self._lowered_flag()

        # otherwise, we will check the new config against the reference config
        else:
            # try to get the new config value for this chain of keys
            try:
                new_val = self._get_nested_value(self._new_config, keys)
            except KeyError:
                raise KeyError("This chain of keys is not in your new config!")

            # try to get the reference config value for this chain of keys
            try:
                ref_val = self._get_nested_value(self._ref_config, keys)
            # if the chain of keys isn't in the reference, we default to None
            except KeyError:
                ref_val = None

            # compare the config values...
            # if they're the same, lower the flag
            if new_val == ref_val:
                flag_file = self._lowered_flag()
            # if something has changed, raise the flag
            else:
                flag_file = self._raised_flag(keys)

        # now, create the flag file
        Path(str(flag_file)).touch()
        # and return the flag to the snakemake rule input
        return flag_file

    def lower_all(self):
        """Lower all the flags, regardless of config changes."""
        self._all_lowered = True
        self._all_raised = False

    def raise_all(self):
        """Raise all the flags, regardless of config changes."""
        self._all_raised = True
        self._all_lowered = False

    def cleanup(self):
        """Remove all the flag files.

        Note this will automatically be called by the __del__ method during
        garbage collection. You likely don't need to explicitly call it yourself.
        """
        rmtree(self._flag_path)

    def __del__(self):
        """Call cleanup() during garbage collection."""
        self.cleanup()


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

    # create the output directory if it doesn't already exist
    output_dir = Path(output_path).parent
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # save the config
    with open(output_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)
