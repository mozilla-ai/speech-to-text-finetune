from transformers import WhisperFeatureExtractor, WhisperTokenizer

from speech_to_text_finetune.data_process import load_local_dataset, process_dataset


def test_process_local_dataset(example_data):
    model_id = "openai/whisper-tiny"

    tokenizer = WhisperTokenizer.from_pretrained(
        model_id, language="English", task="transcribe"
    )

    dataset = load_local_dataset(dataset_dir=example_data, train_split=0.5)

    result = process_dataset(
        dataset,
        feature_extractor=WhisperFeatureExtractor.from_pretrained(model_id),
        tokenizer=tokenizer,
    )

    assert len(dataset["train"]) == len(result["train"])
    assert len(dataset["test"]) == len(result["test"])

    train_tokenized_label_first = result["train"][0]["labels"]
    test_tokenized_label_last = result["test"][-1]["labels"]
    train_text_label_first = tokenizer.decode(
        train_tokenized_label_first, skip_special_tokens=True
    )
    test_text_label_last = tokenizer.decode(
        test_tokenized_label_last, skip_special_tokens=True
    )

    # Make sure the text is being tokenized and indexed correctly
    assert train_text_label_first == dataset["train"][0]["sentence"]
    assert test_text_label_last == dataset["test"][-1]["sentence"]

    # Sample a few transformed audio values and make sure they are in a reasonable range
    assert -100 < result["train"][0]["input_features"][0][10] < 100
    assert -100 < result["train"][0]["input_features"][0][-1] < 100
    assert -100 < result["test"][-1]["input_features"][-1][10] < 100
    assert -100 < result["test"][-1]["input_features"][-1][-1] < 100
