import os
from pathlib import Path

from huggingface_hub.errors import HFValidationError

from speech_to_text_finetune.config import PROC_DATASET_DIR

import pandas as pd
import torch
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
)
from datasets import load_dataset, DatasetDict, Audio, Dataset, load_from_disk
from loguru import logger


def try_find_processed_version(
    dataset_id: str, language_id: str | None = None
) -> DatasetDict | None:
    """
    Try to load a processed version of the dataset if it exists locally. Check if:
    1. The dataset_id is a local path to an already processed dataset directory.
    or
    2. The dataset_id is a path to a local dataset, but a processed version already exists locally.
    or
    3. The dataset_id is a HuggingFace dataset ID, but a processed version already exists locally.
    """
    if Path(dataset_id).name == PROC_DATASET_DIR and Path(dataset_id).is_dir():
        if (
            Path(dataset_id + "/train").is_dir()
            and Path(dataset_id + "/test").is_dir()
            and Path(dataset_id + "/dataset_dict.json").is_file()
        ):
            return load_from_disk(dataset_id)
        else:
            raise FileNotFoundError("Processed dataset is incomplete.")

    proc_dataset_path = _get_local_proc_dataset_path(dataset_id)
    if Path(proc_dataset_path).is_dir():
        return load_from_disk(proc_dataset_path)

    hf_proc_dataset_path = _get_hf_proc_dataset_path(dataset_id, language_id)
    if Path(hf_proc_dataset_path).is_dir():
        logger.info(
            f"Found processed dataset version at {hf_proc_dataset_path} of HF dataset {dataset_id}. "
            f"Loading it directly and skipping processing again the original version."
        )
        return load_from_disk(hf_proc_dataset_path)

    return None


def _get_hf_proc_dataset_path(dataset_id: str, language_id: str) -> str:
    return (
        f"./artifacts/{language_id}_{dataset_id.replace('/', '_')}/{PROC_DATASET_DIR}"
    )


def _get_local_proc_dataset_path(dataset_id: str) -> str:
    return Path(dataset_id).resolve() / PROC_DATASET_DIR


def load_dataset_from_dataset_id(
    dataset_id: str,
    language_id: str | None = None,
    local_train_split: float | None = 0.8,
) -> Tuple[DatasetDict, str]:
    """
    This function loads a dataset, based on the dataset_id and the content of its directory (if it is a local path).
    Possible cases:
    1. The dataset_id is a path to a local, Common Voice dataset directory.

    2. The dataset_id is a path to a local, custom dataset directory.

    3. The dataset_id is a HuggingFace dataset ID.

    Args:
        dataset_id: Path to a processed dataset directory or local dataset directory or HuggingFace dataset ID.
        language_id (Only used for the HF dataset case): Language identifier for the dataset (e.g., 'en' for English)
        local_train_split: (Only used for local datasets) Percentage split of train/test sets

    Returns:
        DatasetDict: A processed dataset ready for training with train/test splits
        str: Path to save the processed directory

    Raises:
        ValueError: If the dataset cannot be found locally or on HuggingFace
    """
    try:
        dataset = _load_local_common_voice(dataset_id, train_split=local_train_split)
        return dataset, _get_local_proc_dataset_path(dataset_id)
    except FileNotFoundError:
        pass

    try:
        dataset = _load_custom_dataset(dataset_id, train_split=local_train_split)
        return dataset, _get_local_proc_dataset_path(dataset_id)
    except FileNotFoundError:
        pass

    try:
        dataset = _load_hf_common_voice(dataset_id, language_id)
        return dataset, _get_hf_proc_dataset_path(dataset_id, language_id)
    except HFValidationError:
        pass
    except FileNotFoundError:
        pass

    raise ValueError(
        f"Could not find dataset {dataset_id}, neither locally nor at HuggingFace. "
        f"If its a private repo, make sure you are logged in locally."
    )


def _load_hf_common_voice(dataset_id: str, language_id: str) -> DatasetDict:
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
        dataset_id,
        language_id,
        split="train+validation",
        trust_remote_code=True,
    )
    common_voice["test"] = load_dataset(
        dataset_id,
        language_id,
        split="test",
        trust_remote_code=True,
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


def _load_local_common_voice(cv_data_dir: str, train_split: float = 0.8) -> DatasetDict:
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
    sentence_map = dict(zip(validated_df["sentence_id"], validated_df["sentence"]))

    # Filter out the rows that don't have a corresponding sentence_id in the sentence_map
    other_df = other_df[other_df["sentence_id"].isin(sentence_map)]

    # Write the full audio clip path
    other_df["audio_clip_path"] = other_df["path"].apply(
        lambda p: cv_data_dir / "clips" / p
    )

    dataset_df = pd.DataFrame(
        {
            "index": other_df.index,
            "sentence": other_df["sentence_id"].map(
                lambda i: sentence_map[i].replace('"', "")
            ),  # remove " characters
            "audio": other_df["audio_clip_path"].astype(str),
            "sentence_id": other_df["sentence_id"],
        }
    )

    train_index = round(len(dataset_df) * train_split)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(dataset_df.iloc[:train_index]),
            "test": Dataset.from_pandas(dataset_df.iloc[train_index:]),
        }
    )

    return dataset


def _load_custom_dataset(dataset_dir: str, train_split: float = 0.8) -> DatasetDict:
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


def load_subset_of_dataset(dataset: Dataset, n_samples: int) -> Dataset:
    return dataset.select(range(n_samples)) if n_samples != -1 else dataset


def process_dataset(
    dataset: DatasetDict,
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
    proc_dataset_path: str,
) -> DatasetDict:
    """
    Process dataset to the expected format by a Whisper model and then save it locally for future use.
    A flag file is also saved in that directory that will be used in future runs to check if the dataset
    is already processed.
    """
    # Create a new column that consists of the resampled audio samples in the right sample rate for whisper
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    dataset = dataset.map(
        _process_inputs_and_labels_for_whisper,
        fn_kwargs={"feature_extractor": feature_extractor, "tokenizer": tokenizer},
        remove_columns=dataset.column_names["train"],
        num_proc=1,
    )

    proc_dataset_path = Path(proc_dataset_path)
    Path.mkdir(proc_dataset_path, parents=True, exist_ok=True)
    dataset.save_to_disk(proc_dataset_path)
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
