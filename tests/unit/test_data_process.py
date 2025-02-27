import pytest
from unittest.mock import patch, MagicMock
from datasets import DatasetDict, Dataset
from speech_to_text_finetune.data_process import load_common_voice, load_local_dataset


@pytest.fixture
def mock_load_dataset():
    with patch("speech_to_text_finetune.data_process.load_dataset") as mocked_load:
        mocked_load.side_effect = [MagicMock(spec=Dataset), MagicMock(spec=Dataset)]
        yield mocked_load


def test_load_common_voice(mock_load_dataset):
    dataset_id, language_id = "mozilla-foundation/common_voice_17_0", "en"
    result = load_common_voice(dataset_id, language_id)

    assert isinstance(result, DatasetDict)
    assert "train" in result
    assert "test" in result
    assert "gender" not in result["train"].features

    mock_load_dataset.assert_any_call(
        dataset_id, language_id, split="train+validation", trust_remote_code=True
    )
    mock_load_dataset.assert_any_call(
        dataset_id, language_id, split="test", trust_remote_code=True
    )


def test_load_local_dataset_default_split(example_data):
    dataset = load_local_dataset(dataset_dir=example_data)

    assert len(dataset["train"]) == 8
    assert len(dataset["test"]) == 2

    assert dataset["train"][0]["sentence"] == "GO DO YOU HEAR"
    assert dataset["train"][0]["audio"] == f"{example_data}/rec_0.wav"

    assert dataset["test"][-1]["sentence"] == "DO YOU KNOW THE ASSASSIN ASKED MORREL"
    assert dataset["test"][-1]["audio"] == f"{example_data}/rec_9.wav"


def test_load_local_dataset_no_test(example_data):
    dataset = load_local_dataset(dataset_dir=example_data, train_split=1.0)

    assert len(dataset["train"]) == 10
    assert len(dataset["test"]) == 0
