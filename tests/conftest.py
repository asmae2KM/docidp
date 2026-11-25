import shutil
from pathlib import Path

import pytest

from evaluation.generate_dataset import generate

_real_which = shutil.which


@pytest.fixture(autouse=True)
def _no_real_marker(monkeypatch):
    """A locally installed marker-pdf (minutes per PDF on CPU) must never leak into the tests."""
    monkeypatch.setattr(
        shutil, "which", lambda name, *a, **k: None if name == "marker_single" else _real_which(name, *a, **k)
    )


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("dataset")
    generate(out, seed=42)
    return out
