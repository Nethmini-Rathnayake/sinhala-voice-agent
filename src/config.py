"""Configuration loading.

``load_config()`` reads ``config.yaml`` from the repo root (cached).
``select_device()`` and ``hf_cache_dir()`` resolve runtime settings.
``load_env()`` loads a local ``.env`` file into the environment if one exists;
in Colab, keys are set from ``google.colab.userdata`` by the notebook instead.

CLI: ``python -m src.config`` prints the loaded config.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


@lru_cache(maxsize=None)
def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Return the parsed config.yaml as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_device() -> str:
    """Return "cuda" or "cpu" according to ``device.prefer`` (auto/cuda/cpu)."""
    import torch

    prefer = load_config()["device"]["prefer"]
    if prefer == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if prefer == "cuda":
        raise RuntimeError("device.prefer is 'cuda' but CUDA is not available")
    return "cpu"


def hf_cache_dir() -> Path:
    """Model download cache: ``hf.cache_dir`` in Colab, ``hf.local_cache_dir`` elsewhere."""
    hf = load_config()["hf"]
    colab_dir = Path(hf["cache_dir"])
    if colab_dir.parent.exists():
        return colab_dir
    return REPO_ROOT / hf["local_cache_dir"]


def load_env() -> None:
    """Load ``.env`` from the repo root without overriding existing variables."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


if __name__ == "__main__":
    import pprint

    pprint.pprint(load_config())
