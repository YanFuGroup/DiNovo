# coding: utf-8
"""Find the path to LightGBM dynamic library files."""
from os import path
from typing import List


def find_lib_path() -> List[str]:
    """Find the path to LightGBM library files.

    Returns
    -------
    lib_path: list of str
       List of all found library paths to LightGBM."""
    return [path.dirname(__file__) + "\\lib_lightgbm.dll"]
