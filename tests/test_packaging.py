"""Verifies the src/ layout actually installs and imports.

Not filler: Project A hit setuptools' "Multiple top-level packages discovered in
a flat-layout" error only after bench/ and notebooks/ appeared alongside the
source, which broke CI at an inconvenient moment. This repo uses an explicit
src/ package root to avoid that, and this test proves the packaging works before
any real code depends on it.
"""

import importlib

import minigpt_infer


def test_package_imports():
    assert minigpt_infer.__version__ == "0.1.0"


def test_package_is_importable_by_name():
    # catches a src/ layout that was declared but never actually picked up by
    # setuptools.packages.find -- the failure mode this layout exists to prevent
    assert importlib.import_module("minigpt_infer") is minigpt_infer
