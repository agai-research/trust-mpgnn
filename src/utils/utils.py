"""
utils.py - Utility functions: logging, path resolution, JSON I/O.
Author: H. Mezni
"""

import os
import json
import logging
import time

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(log_dir: str = "logs", level: int = logging.INFO):
    """Configure file + console logging."""
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{ts}.log")
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_config(path: str = "config.json") -> dict:
    return load_json(path)


def resolve_path(base: str, rel: str) -> str:
    """Resolve a relative path from a base directory."""
    return os.path.normpath(os.path.join(base, rel))
