import yaml
import numpy as np
from pathlib import Path

def load_config(section=None, filename="optimization_config.yaml"):
    """
    Load configuration from a YAML file, with optional section selection.

    Parameters
    ----------
    section : str, optional
        Specific section to return (e.g. "mechanical", "options").
        If None, returns the full config.
    filename : str, optional
        Name of the YAML file to load. Defaults to "optimization_config.yaml".

    Returns
    -------
    dict
        Configuration dictionary (optionally limited to a single section).
    """
    # Resolve file path relative to this module’s directory
    config_path = Path(__file__).parent / filename
    if not config_path.exists():
        raise FileNotFoundError(f"[load_config] Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Recursively convert numeric lists → numpy arrays
    def convert_lists_to_arrays(obj):
        if isinstance(obj, dict):
            return {k: convert_lists_to_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list) and all(isinstance(x, (int, float)) for x in obj):
            return np.array(obj, dtype=float)
        return obj

    config = convert_lists_to_arrays(config)

    # Return specific section or full config
    if section:
        return config.get(section, {})
    return config

