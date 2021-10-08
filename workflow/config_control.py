"""Tools to help handle snakemake configs."""
from pathlib import Path
from shutil import rmtree
from typing import List, Union, Callable
from warnings import warn
from itertools import count

import yaml
from snakemake.io import AnnotatedString, ancient


class ConfigFlags:
    """Controls flagging the config file so that if sections of the config
    are changed, the corresponding rules will be rerun.

    Note you can only flag the top-level keys in config.
    """

    def __init__(
        self, new_config: dict, old_config_path: str, default_config_path: str = None
    ):
        """
        Parameters
        ----------
        new_config : dict
            The dictionary of new config settings
        old_config_path : str
            The path to the old config file (or the path where the new config
            file will be later saved).
        default_config_path : str, optional
            The path to the default config file against which to check the
            new config, if there is no config at old_config_path
        """

        # create an array to hold all the flags
        self._all_flags = []

        # set the path where the flags will be stored
        self._flag_dir = Path(old_config_path).parent / ".config_flags"

        # setup the flag dictionary
        # for each key, we will store
        #   - the path to the flags
        #   - whether or not the key has been flagged (i.e. if the config
        #       for this key has changed)
        #   - whether or not a snakemake rule has used this flag
        self._flag_dict = {
            key: {
                "path": f"{self._flag_dir}/{key}",
                "iter": count(),
                "flagged": True,
                "used": False,
            }
            for key in new_config
        }

        # now we will create directories for the flags
        for val in self._flag_dict.values():
            flag_path = val["path"]
            Path(flag_path).mkdir(parents=True, exist_ok=True)

        # set the reference config
        ref_config = None

        # if the old config exists, use that as the reference config
        if Path(old_config_path).is_file():
            with open(old_config_path, "r") as file:
                ref_config = yaml.safe_load(file)
        # else if a default config is passed, used that as the ref config
        elif default_config_path is not None:
            with open(default_config_path, "r") as file:
                ref_config = yaml.safe_load(file)

        # if we have a reference config, compare the config for each key
        # of the new config against the reference, and if they are the same
        # lower the flag for that key so the corresponding tules are not
        # re-run by snakemake
        if ref_config is not None:
            for key, val in self._flag_dict.items():
                if new_config[key] == ref_config[key]:
                    val["flagged"] = False

    def flag(self, key: str) -> Union[str, AnnotatedString, List[AnnotatedString]]:
        """Set a flag for the given config key in a snakemake rule input."""

        # note that a snakemake rule has used this flag
        self._flag_dict[key]["used"] = True

        # get the flag file
        flag_path = self._flag_dict[key]["path"]
        flag_iter = self._flag_dict[key]["iter"]
        flag_file = f"{flag_path}/flag{next(flag_iter)}"

        # if the key is flagged, note that this flag is raised, and return a
        # normal string, so that snakemake will consider the flag newer than
        # the output and the corresponding rule will be triggered
        if self._flag_dict[key]["flagged"]:
            flag_file += "_raised"
        # else, if the key isn't flagged, note that this flag is lowered, and
        # return an ancient string, so that snakemake will consider the flag
        # older than the output and the corresponding rule will not be triggered
        else:
            flag_file += "_lowered"
            flag_file = ancient(flag_file)

        # create the flag
        Path(str(flag_file)).touch()

        # append the flag to the list of all flags
        self._all_flags.append(flag_file)

        # return the flag to the input of a snakemake rule
        return flag_file

    def all_flags(self, wildcards=None) -> Callable:
        """Return all the flags for input to the snakemake all rule."""
        return lambda wildcards: self._all_flags

    def cleanup(self):
        """Remove all the flag files."""

        # iterate through all the keys
        for key, val in self._flag_dict.items():
            # if the key is flagged, but the flag isn't used, send a warning
            if val["flagged"] is True and val["used"] is False:
                warn(
                    "\n\nWARNING: "
                    f"Config for {key} changed, but this didn't trigger any rules. "
                    "You likely need to flag the rules in the Snakefile that use "
                    "this set of config settings.\n",
                    stacklevel=7,
                )

            # remove the flags
            rmtree(val["path"])

        # delete the folder that held all the flags
        Path(self._flag_dir).rmdir()


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
