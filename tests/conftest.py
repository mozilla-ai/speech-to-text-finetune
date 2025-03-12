from pathlib import Path

import pytest

from speech_to_text_finetune.data_process import load_dataset_from_dataset_id


@pytest.fixture(scope="session")
def example_config_path():
    return str(Path(__file__).parent.parent / "tests/e2e/config.yaml")


@pytest.fixture(scope="session")
def custom_data_path():
    return str(Path(__file__).parent.parent / "example_data/custom")


@pytest.fixture(scope="session")
def local_common_voice_data_path():
    return str(
        Path(__file__).parent.parent / "example_data/example_cv_dataset/language_id/"
    )


@pytest.fixture(scope="session")
def custom_dataset_half_split(custom_data_path):
    return load_dataset_from_dataset_id(
        dataset_id=custom_data_path, local_train_split=0.5
    )
