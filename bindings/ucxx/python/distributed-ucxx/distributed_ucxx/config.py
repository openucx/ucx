# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration management for distributed-ucxx.

This module provides configuration loading and validation for the distributed-ucxx
package, replacing the UCX configuration previously handled by the main distributed
package.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

import dask


def _load_default_config() -> Dict[str, Any]:
    """Load the default configuration from distributed-ucxx.yaml."""
    config_path = Path(__file__).parent / "distributed-ucxx.yaml"

    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    # Fallback default configuration
    return {
        "distributed-ucxx": {
            "version": 1,
            "cuda-copy": None,
            "tcp": None,
            "nvlink": None,
            "infiniband": None,
            "rdmacm": None,
            "create-cuda-context": None,
            "multi-buffer": None,
            "environment": {},
            "rmm": {
                "pool-size": None,
            },
        }
    }


def get_ucx_config(key: str, default: Any = None) -> Any:
    """
    Get a UCX configuration value.

    Parameters
    ----------
    key : str
        Configuration key under 'distributed-ucxx'
    default : Any, optional
        Default value if key is not found

    Returns
    -------
    Any
        Configuration value
    """
    # First try the new distributed-ucxx namespace
    full_key = f"distributed-ucxx.{key}"
    value = dask.config.get(full_key, default=None)

    if value is not None:
        return value

    # Fallback to legacy distributed namespace for backward compatibility
    legacy_key = f"distributed.comm.ucx.{key}"
    return dask.config.get(legacy_key, default=default)


def get_rmm_config(key: str, default: Any = None) -> Any:
    """
    Get an RMM configuration value.

    Parameters
    ----------
    key : str
        RMM configuration key (e.g., 'pool-size')
    default : Any, optional
        Default value if key is not found

    Returns
    -------
    Any
        Configuration value
    """
    # First try the new distributed-ucxx namespace
    full_key = f"distributed-ucxx.rmm.{key}"
    value = dask.config.get(full_key, default=None)

    if value is not None:
        return value

    # Fallback to legacy distributed namespace for backward compatibility
    legacy_key = f"distributed.rmm.{key}"
    return dask.config.get(legacy_key, default=default)


def _flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary using dot notation.

    Parameters
    ----------
    d : dict
        Dictionary to flatten
    parent_key : str
        Parent key prefix
    sep : str
        Separator to use between keys

    Returns
    -------
    dict
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def setup_config() -> None:
    """
    Set up distributed-ucxx configuration.

    This should be called during module initialization to ensure the
    default configuration is loaded into dask.config.
    """
    default_config = _load_default_config()

    # Only set defaults if they don't already exist
    for key, value in _flatten_dict(default_config).items():
        if dask.config.get(key, default=None) is None:
            dask.config.set({key: value})
