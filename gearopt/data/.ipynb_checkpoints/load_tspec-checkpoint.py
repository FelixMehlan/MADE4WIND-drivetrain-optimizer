import numpy as np
import pandas as pd
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def load_tspec(filepath: str):
    """
    Load torque specification (T_spec) from CSV as a NumPy array.
    Uses an in-memory cache so the file is read only once per session.
    """
    filepath = os.path.normpath(filepath)

    if not os.path.isabs(filepath):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        filepath = os.path.normpath(os.path.join(base_dir, filepath))

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[load_tspec] File not found: {filepath}")

    try:
        df = pd.read_csv(filepath, header=None, encoding="utf-8-sig")
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

        if df.shape[0] == 0 or df.shape[1] == 0:
            print(f"[load_tspec] No numeric data found in {filepath}")
            return np.array([[1.0, 1.0, 1.0]])

        return df.to_numpy(dtype=float)

    except Exception as e:
        print(f"[load_tspec] Failed to load {filepath}: {e}")
        return np.array([[1.0, 1.0, 1.0]])
