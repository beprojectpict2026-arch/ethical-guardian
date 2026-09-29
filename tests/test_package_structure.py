"""Guards the package layout: every public module must import cleanly."""

import importlib
from importlib import resources

import pytest

import eguard

MODULES = [
    "eguard.results",
    "eguard.metrics",
    "eguard.metrics.inequality",
    "eguard.metrics.fairness",
    "eguard.metrics.labour",
    "eguard.metrics.market",
    "eguard.manifest",
    "eguard.manifest.schema",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)


def test_version_is_set():
    assert eguard.__version__ != "0.0.0", "package metadata not found; run `uv sync`"


def test_package_ships_type_marker():
    assert resources.files("eguard").joinpath("py.typed").is_file()
