import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config["monitoring"]["samples_per_check"] < 1:
        raise ValueError("samples_per_check must be at least 1")
    return config
