import os
from pathlib import Path

import pandas as pd
import torch
from dataclasses import dataclass
from typing import Dict, List, Union

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
)

from datasets import load_dataset, DatasetDict, Audio, Dataset


def load_common_voice(dataset_id: str, language_id: str) -> DatasetDict:
    """
    Load the default train+validation split used for finetuning and a test split used for evaluation.
    Args:
        dataset_id: official Common Voice dataset id from the mozilla-foundation organisation from Hugging Face
        language_id: a registered language identifier from Common Voice (most often in ISO-639 format)

    Returns:
        DatasetDict: HF Dataset dictionary that consists of two distinct Datasets
    """
    common_voice = DatasetDict()

    common_voice["train"] = load_dataset(
        dataset_id, language_id, split="train+validation", trust_remote_code=True
    )
    common_voice["test"] = load_dataset(
        dataset_id, language_id, split="test", trust_remote_code=True
    )
    common_voice = common_voice.remove_columns(
        [
            "accent",
            "age",
            "client_id",
            "down_votes",
            "gender",
            "locale",
            "path",
            "segment",
            "up_votes",
        ]
    )

    return common_voice


def load_local_common_voice(cv_data_dir: str, train_split: float = 0.8) -> DatasetDict:
    """
    Load a local Common Voice dataset (as downloaded from the official Common Voice website) into a DatasetDict.
    We only use the validated.tsv file to source the data to use for both training and testing.

    Args:
        cv_data_dir (str): path to the local Common Voice dataset directory
        train_split (str): percentage split of the dataset to train+validation and test set

    Returns:
        DatasetDict: HF Dataset dictionary that consists of two distinct Datasets (train+validation and test)
    """
    cv_data_dir = Path(cv_data_dir)
    validated_df = pd.read_csv(cv_data_dir / "validated_sentences.tsv", sep="\t")
    other_df = pd.read_csv(cv_data_dir / "other.tsv", sep="\t")

    # Map sentence_id to sentences to then use the sentence_id to pull the correct audio path from other.tsv
    sentence_map = {
        row["sentence_id"]: row["sentence"] for _, row in validated_df.iterrows()
    }

    local_dataset = []

    for i, row in other_df.iterrows():
        sentence_id = row["sentence_id"]
        audio_clip_path = cv_data_dir / "clips" / row["path"]

        if sentence_id in sentence_map and Path(audio_clip_path).exists():
            # Set rows: index, string without any " characters, the original audio clip path and the sentence id
            local_dataset.append(
                {
                    "index": i,
                    "sentence": sentence_map[sentence_id].replace('"', ""),
                    "audio": str(audio_clip_path),
                    "sentence_id": sentence_id,
                }
            )

    dataset_df = pd.DataFrame(local_dataset)
    train_index = round(len(dataset_df) * train_split)

    dataset = DatasetDict()
    dataset["train"] = Dataset.from_pandas(dataset_df[:train_index])
    dataset["test"] = Dataset.from_pandas(dataset_df[train_index:])

    return dataset


def load_local_dataset(dataset_dir: str, train_split: float = 0.8) -> DatasetDict:
    """
    Load sentences and accompanied recorded audio files into a pandas DataFrame, then split into train/test and finally
    load it into two distinct train Dataset and test Dataset.

    Sentences and audio files should be indexed like this: <index>: <sentence> should be accompanied by rec_<index>.wav

    Args:
        dataset_dir (str): path to the local dataset, expecting a text.csv and .wav files under the directory
        train_split (float): percentage split of the dataset to train+validation and test set

    Returns:
        DatasetDict: HF Dataset dictionary that consists of two distinct Datasets (train+validation and test)
    """
    text_file = dataset_dir + "/text.csv"

    dataframe = pd.read_csv(text_file)
    audio_files = sorted(
        [
            f"{dataset_dir}/{f}"
            for f in os.listdir(dataset_dir)
            if f.endswith(".wav") or f.endswith(".mp3")
        ],
    )

    dataframe["audio"] = audio_files
    train_index = round(len(dataframe) * train_split)

    my_data = DatasetDict()
    my_data["train"] = Dataset.from_pandas(dataframe[:train_index])
    my_data["test"] = Dataset.from_pandas(dataframe[train_index:])

    return my_data


def process_dataset(
    dataset: DatasetDict,
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
) -> DatasetDict:
    """
    Process dataset to the expected format by a Whisper model.
    """
    # Create a new column that consists of the resampled audio samples in the right sample rate for whisper
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    dataset = dataset.map(
        _process_inputs_and_labels_for_whisper,
        fn_kwargs={"feature_extractor": feature_extractor, "tokenizer": tokenizer},
        remove_columns=dataset.column_names["train"],
        num_proc=1,
    )
    return dataset


def _process_inputs_and_labels_for_whisper(
    batch: Dict, feature_extractor: WhisperFeatureExtractor, tokenizer: WhisperTokenizer
) -> Dict:
    """
    Use Whisper's feature extractor to transform the input audio arrays into log-Mel spectrograms
     and the tokenizer to transform the text-label into tokens. This function is expected to be called using
     the .map method in order to process the data batch by batch.
    """
    audio = batch["audio"]

    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data Collator class in the format expected by Seq2SeqTrainer used for processing
    input data and labels in batches while finetuning. More info here:
    """

    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if labels already have a bos token, remove it since its appended later
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
