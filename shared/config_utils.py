import os
import yaml

def read_yaml_config(yml_file=None):
    """Read YAML config and return dictionary of items."""
    if yml_file is None:
        yml_file = os.path.join(os.path.dirname(__file__), "config.yml")
    with open(yml_file, "r") as file:
        config = yaml.safe_load(file)
        return config