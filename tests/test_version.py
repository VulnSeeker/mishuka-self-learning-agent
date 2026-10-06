"""Sanity tests for onyx.version."""

from __future__ import annotations

import onyx
from onyx.version import __author__, __license__, __version__


def test_version_format():
    assert isinstance(__version__, str)
    assert __version__.count(".") >= 1


def test_metadata_present():
    assert __author__ == "vulnseeker"
    assert __license__ == "MIT"


def test_package_exports_version():
    assert onyx.__version__ == __version__
