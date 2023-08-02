from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """loads a config file from the given path

    Args:
        path (str): path to config file

    Returns:
        Dict[str, Any]: config file content
    """
    with open(path) as f:
        config = yaml.safe_load(f)

    return config


def write_yaml(path: str, content: Dict[str, Any]):
    """writes yaml file to filesystem

    Args:
        path (str): where to store yaml file
        content (Dict[str, Any]): content of yaml file
    """
    with open(path, "w") as file:
            yaml.dump(content, file)