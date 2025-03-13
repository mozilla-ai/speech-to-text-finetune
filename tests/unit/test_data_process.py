from unittest.mock import MagicMock, patch

import pytest
from datasets import DatasetDict, Dataset

from speech_to_text_finetune.data_process import (
    load_dataset_from_dataset_id,
    load_subset_of_dataset,
    try_find_processed_version,
)


@pytest.fixture
def mock_load_hf_dataset():
    with patch("speech_to_text_finetune.data_process.load_dataset") as mocked_load:
        mock_dataset = MagicMock(spec=Dataset)
        mock_dataset.features = {"index": None, "sentence": None, "audio": None}
        mock_dataset.remove_columns = MagicMock(return_value=mock_dataset)
        mocked_load.side_effect = [mock_dataset, mock_dataset]
        yield mocked_load


def test_try_find_processed_version_hf():
    dataset = try_find_processed_version(
        dataset_id="mozilla-foundation/common_voice_17_0", language_id="en"
    )
    assert dataset is None


def _assert_proper_dataset(dataset: DatasetDict) -> None:
    assert isinstance(dataset, DatasetDict)
    assert "index" in dataset["train"].features
    assert "sentence" in dataset["train"].features
    assert "audio" in dataset["train"].features

    assert "index" in dataset["test"].features
    assert "sentence" in dataset["test"].features
    assert "audio" in dataset["test"].features


def test_load_dataset_from_dataset_id_local_cv(local_common_voice_data_path):
    dataset, _ = load_dataset_from_dataset_id(dataset_id=local_common_voice_data_path)
    _assert_proper_dataset(dataset)


def test_load_dataset_from_dataset_id_custom(custom_data_path):
    dataset, _ = load_dataset_from_dataset_id(dataset_id=custom_data_path)
    _assert_proper_dataset(dataset)


def test_load_dataset_from_dataset_id_hf_cv(mock_load_hf_dataset):
    dataset, _ = load_dataset_from_dataset_id(
        dataset_id="mozilla-foundation/common_voice_17_0", language_id="en"
    )
    _assert_proper_dataset(dataset)


def test_load_local_common_voice_split(local_common_voice_data_path):
    dataset, _ = load_dataset_from_dataset_id(
        dataset_id=local_common_voice_data_path, local_train_split=0.5
    )

    assert len(dataset["train"]) == 1
    assert len(dataset["test"]) == 1

    assert dataset["train"][0]["sentence"] == "Example sentence"
    assert (
        dataset["train"][0]["audio"]
        == f"{local_common_voice_data_path}/clips/an_example.mp3"
    )

    assert dataset["test"][-1]["sentence"] == "Another example sentence"
    assert (
        dataset["test"][-1]["audio"]
        == f"{local_common_voice_data_path}/clips/an_example_2.mp3"
    )


def test_load_custom_dataset_default_split(custom_data_path):
    dataset, _ = load_dataset_from_dataset_id(dataset_id=custom_data_path)

    assert len(dataset["train"]) == 8
    assert len(dataset["test"]) == 2

    assert dataset["train"][0]["sentence"] == "GO DO YOU HEAR"
    assert dataset["train"][0]["audio"] == f"{custom_data_path}/rec_0.wav"

    assert dataset["test"][-1]["sentence"] == "DO YOU KNOW THE ASSASSIN ASKED MORREL"
    assert dataset["test"][-1]["audio"] == f"{custom_data_path}/rec_9.wav"


def test_load_custom_dataset_no_test(custom_data_path):
    dataset, _ = load_dataset_from_dataset_id(
        dataset_id=custom_data_path, local_train_split=1.0
    )

    assert len(dataset["train"]) == 10
    assert len(dataset["test"]) == 0


def test_load_subset_of_dataset_train(custom_dataset_half_split):
    subset = load_subset_of_dataset(custom_dataset_half_split["train"], n_samples=-1)

    assert len(subset) == len(custom_dataset_half_split["train"]) == 5

    subset = load_subset_of_dataset(custom_dataset_half_split["train"], n_samples=5)
    assert len(subset) == len(custom_dataset_half_split["train"]) == 5

    subset = load_subset_of_dataset(custom_dataset_half_split["train"], n_samples=2)
    assert len(subset) == 2

    subset = load_subset_of_dataset(custom_dataset_half_split["train"], n_samples=0)
    assert len(subset) == 0

    subset = load_subset_of_dataset(custom_dataset_half_split["test"], n_samples=-1)
    assert len(subset) == len(custom_dataset_half_split["test"]) == 5

    with pytest.raises(IndexError):
        load_subset_of_dataset(custom_dataset_half_split["train"], n_samples=6)
