"""Adapters to the user's existing local projects; no source files are modified."""

import importlib
import os
import sys
from pathlib import Path


def local_module(name: str, project: str, subfolder: str = ""):
    root = Path(os.environ.get("SUPERVARIO_PROJECTS", Path(__file__).resolve().parents[2]))
    directory = root / project / subfolder
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Required local project not found: {directory}. "
            "Set SUPERVARIO_PROJECTS to the folder containing the existing projects."
        )
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    return importlib.import_module(name)


def readers():
    return local_module("srkreliability.io_utils", "SRKReliability")


def filters():
    return local_module("srkreliability.filters", "SRKReliability")


def mesh_tools():
    return local_module("mesh_heal", "MeshOps")


def bmf_tools():
    return local_module("bmf_standalone_exporter", "Anterpolator", "Anterpolator3DViewer")
