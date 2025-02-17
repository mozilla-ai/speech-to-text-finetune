from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def example_data():
    return str(Path(__file__).parent.parent / "example_data/custom")


@pytest.fixture(scope="session")
def language_file_path():
    return str(
        Path(__file__).parent.parent / "example_data/languages_common_voice_17_0.json"
    )
