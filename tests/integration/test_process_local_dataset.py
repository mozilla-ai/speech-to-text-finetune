import shutil
from unittest.mock import patch

import pytest
from datasets import DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from speech_to_text_finetune.config import PROC_DATASET_DIR
from speech_to_text_finetune.data_process import (
    process_dataset,
    try_find_processed_version,
    load_dataset_from_dataset_id,
)


@pytest.fixture
def mock_dataset_map():
    with patch(
        "speech_to_text_finetune.data_process._process_inputs_and_labels_for_whisper"
    ) as mocked_process:
        mocked_process.return_value = {
            "input_features": [0.1, 0.2, 0.3],
            "labels": [1, 2, 3],
        }
        yield mocked_process


@pytest.fixture
def proc_custom_data_path(custom_data_path):
    return f"{custom_data_path}/{PROC_DATASET_DIR}"


@pytest.mark.parametrize(
    "dataset_id",
    ["local_common_voice_data_path", "custom_data_path"],
)
def test_load_proc_dataset_after_init_processing(
    dataset_id,
    request,
    mock_whisper_feature_extractor,
    mock_whisper_tokenizer,
    mock_dataset_map,
):
    # Arguments in the parametrize decorator are fixtures, not actual values
    dataset_id = request.getfixturevalue(dataset_id)

    # First make sure there is no processed dataset version locally
    shutil.rmtree(f"{dataset_id}/{PROC_DATASET_DIR}", ignore_errors=True)
    dataset = try_find_processed_version(dataset_id=dataset_id)
    assert dataset is None

    # Load, process the dataset and save it under proc_dataset_dir
    dataset, proc_dataset_dir = load_dataset_from_dataset_id(
        dataset_id=dataset_id, local_train_split=0.5
    )
    process_dataset(
        dataset=dataset,
        feature_extractor=mock_whisper_feature_extractor,
        tokenizer=mock_whisper_tokenizer,
        proc_dataset_path=proc_dataset_dir,
    )
    # Now try again to find and load the processed version
    dataset = try_find_processed_version(dataset_id=dataset_id)
    assert isinstance(dataset, DatasetDict)

    # Cleanup
    shutil.rmtree(proc_dataset_dir)


def test_process_local_dataset(custom_dataset_half_split, tmp_path):
    model_id = "openai/whisper-tiny"

    tokenizer = WhisperTokenizer.from_pretrained(
        model_id, language="English", task="transcribe"
    )

    result = process_dataset(
        custom_dataset_half_split,
        feature_extractor=WhisperFeatureExtractor.from_pretrained(model_id),
        tokenizer=tokenizer,
        proc_dataset_path=str(tmp_path),
    )

    assert len(custom_dataset_half_split["train"]) == len(result["train"])
    assert len(custom_dataset_half_split["test"]) == len(result["test"])

    train_tokenized_label_first = result["train"][0]["labels"]
    test_tokenized_label_last = result["test"][-1]["labels"]
    train_text_label_first = tokenizer.decode(
        train_tokenized_label_first, skip_special_tokens=True
    )
    test_text_label_last = tokenizer.decode(
        test_tokenized_label_last, skip_special_tokens=True
    )

    # Make sure the text is being tokenized and indexed correctly
    assert train_text_label_first == custom_dataset_half_split["train"][0]["sentence"]
    assert test_text_label_last == custom_dataset_half_split["test"][-1]["sentence"]

    # Sample a few transformed audio values and make sure they are in a reasonable range
    assert -100 < result["train"][0]["input_features"][0][10] < 100
    assert -100 < result["train"][0]["input_features"][0][-1] < 100
    assert -100 < result["test"][-1]["input_features"][-1][10] < 100
    assert -100 < result["test"][-1]["input_features"][-1][-1] < 100
