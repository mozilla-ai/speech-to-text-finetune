import pytest
from unittest.mock import patch, MagicMock
from datasets import DatasetDict, Dataset
from transformers import WhisperTokenizer, WhisperFeatureExtractor

from speech_to_text_finetune.data_process import (
    load_dataset_from_dataset_id,
    load_subset_of_dataset,
)


@pytest.fixture
def mock_load_dataset_from_datasets_package():
    with patch("speech_to_text_finetune.data_process.load_dataset") as mocked_load:
        mocked_load.side_effect = [MagicMock(spec=Dataset), MagicMock(spec=Dataset)]
        yield mocked_load


@pytest.fixture
def mock_whisper_feature_extractor():
    return MagicMock(spec=WhisperFeatureExtractor)


@pytest.fixture
def mock_whisper_tokenizer():
    return MagicMock(spec=WhisperTokenizer)


@pytest.mark.parametrize(
    "dataset_id, language_id",
    [
        ("local_common_voice_data_path", None),
        ("custom_data_path", None),
        ("mozilla-foundation/common_voice_17_0", "en"),
    ],
)
def test_load_dataset_from_dataset_id(
    dataset_id: str,
    language_id: str | None,
):
    dataset = load_dataset_from_dataset_id(
        dataset_id=dataset_id,
        language_id=language_id,
    )

    assert isinstance(dataset, DatasetDict)
    assert "train" in dataset
    assert "test" in dataset
    assert "gender" not in dataset["train"].features
    assert "index" in dataset["train"]
    assert "sentence" in dataset["train"]
    assert "audio" in dataset["train"]
    assert "sentence_id" in dataset["train"]


def test_load_local_common_voice_split(local_common_voice_data_path):
    dataset = load_dataset_from_dataset_id(
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
    dataset = load_dataset_from_dataset_id(dataset_id=custom_data_path)

    assert len(dataset["train"]) == 8
    assert len(dataset["test"]) == 2

    assert dataset["train"][0]["sentence"] == "GO DO YOU HEAR"
    assert dataset["train"][0]["audio"] == f"{custom_data_path}/rec_0.wav"

    assert dataset["test"][-1]["sentence"] == "DO YOU KNOW THE ASSASSIN ASKED MORREL"
    assert dataset["test"][-1]["audio"] == f"{custom_data_path}/rec_9.wav"


def test_load_custom_dataset_no_test(custom_data_path):
    dataset = load_dataset_from_dataset_id(
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
