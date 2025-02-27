from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def example_data():
    return str(Path(__file__).parent.parent / "example_data/custom")
