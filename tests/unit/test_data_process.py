from unittest.mock import MagicMock, patch

import pytest
from datasets import DatasetDict, Dataset

from speech_to_text_finetune.data_process import (
    load_dataset_from_dataset_id,
    load_subset_of_dataset,
    try_find_processed_version,
    process_dataset,
)


@pytest.fixture
def mock_load_hf_dataset():
    with patch("speech_to_text_finetune.data_process.load_dataset") as mocked_load:
        mock_dataset = MagicMock(spec=Dataset)
        mock_dataset.features = {"index": None, "sentence": None, "audio": None}
        mock_dataset.select_columns = MagicMock(return_value=mock_dataset)
        mocked_load.side_effect = [mock_dataset, mock_dataset]
        yield mocked_load


def test_try_find_processed_version_hf():
    dataset = try_find_processed_version(
        dataset_id="mozilla-foundation/common_voice_17_0", language_id="en"
    )
    assert dataset is None


def _assert_proper_dataset(dataset: DatasetDict) -> None:
    assert isinstance(dataset, DatasetDict)
    assert "sentence" in dataset["train"].features
    assert "audio" in dataset["train"].features

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


@pytest.fixture
def mock_dataset():
    data = {
        "audio": [
            {"array": [0.0] * 16000 * 31, "sampling_rate": 16000},  # 31 seconds
            {"array": [0.0] * 16000 * 29, "sampling_rate": 16000},  # 29 seconds
            {"array": [0.0] * 16000 * 29, "sampling_rate": 16000},  # 29 seconds
        ],
        "sentence": [
            "This is an invalid audio sample.",
            "This is a valid audio sample.",
            "This is a really long text. So long that its actually impossible for Whisper to fully generate such a "
            "long text, meaning that this text should be removed from the dataset. Yeap. Exactly. Completely removed."
            "But actually, because we are mocking the processor, and we are just returning as tokenized labels, this"
            "text itself as-is (see how mock_whisper_processor is implemented), its this text itself that needs to be "
            "longer than 448 (the max generation length of whisper) not the tokenized version of it.",
        ],
    }
    return DatasetDict({"train": Dataset.from_dict(data)})


def test_remove_long_audio_and_transcription_samples(
    mock_dataset, mock_whisper_processor, tmp_path
):
    processed_dataset = process_dataset(
        dataset=mock_dataset,
        processor=mock_whisper_processor,
        batch_size=1,
        proc_dataset_path=str(tmp_path),
    )
    assert len(processed_dataset["train"]) == 1
    assert processed_dataset["train"][0]["sentence"] == "This is a valid audio sample."
