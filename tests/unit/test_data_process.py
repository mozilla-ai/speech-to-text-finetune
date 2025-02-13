import pytest
from unittest.mock import patch, MagicMock
from datasets import DatasetDict, Dataset
from speech_to_text_finetune.data_process import load_common_voice


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

    mock_load_dataset.assert_any_call(dataset_id, language_id, split="train+validation")
    mock_load_dataset.assert_any_call(dataset_id, language_id, split="test")
