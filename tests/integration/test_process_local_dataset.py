from transformers import WhisperFeatureExtractor, WhisperTokenizer

from speech_to_text_finetune.data_process import process_dataset


def test_load_processed_dataset(custom_dataset_half_split):
    return


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
