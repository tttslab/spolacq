import os
import yaml
from pathlib import Path


class HParams:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def parse(self, hparams_string: str) -> None:
        if not hparams_string:
            return

        for param in hparams_string.split(","):
            key, value = param.split("=")
            key = key.strip()
            value = value.strip()

            try:
                if "." in value:
                    value = float(value)
                elif value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                else:
                    value = int(value)
            except ValueError:
                pass  # If the value cannot be converted, it remains as a string.

            setattr(self, key, value)

    def values(self):
        return self._hparams


def create_hparams(config, data_file_paths: list[Path]):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams_dict = {}
    # Flatten the parameters for each section
    print("config:", config)
    for section_name, section_params in config.items():
        if isinstance(section_params, dict):
            for param_name, param_value in section_params.items():
                hparams_dict[param_name] = param_value
        else:
            hparams_dict[section_name] = section_params

    # The file path is obtained from config.
    hparams_dict["training_files"] = data_file_paths["train_path"]
    hparams_dict["validation_files"] = data_file_paths["val_path"]

    hparams = HParams(**hparams_dict)
    print("Hyperparameters:", hparams.__dict__)
    return hparams
