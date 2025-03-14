from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def example_config():
    return str(Path(__file__).parent.parent / "tests/e2e/config.yaml")


@pytest.fixture(scope="session")
def example_custom_data():
    return str(Path(__file__).parent.parent / "example_data/custom")


@pytest.fixture(scope="session")
def example_common_voice_data():
    return str(
        Path(__file__).parent.parent / "example_data/example_cv_dataset/language_id/"
    )
