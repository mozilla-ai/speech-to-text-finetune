import os
from pathlib import Path
from speech_to_text_finetune.config import PROC_DATASET_DIR

import pandas as pd
import torch
from dataclasses import dataclass
from typing import Dict, List, Union

from huggingface_hub import repo_exists
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
)
from datasets import load_dataset, DatasetDict, Audio, Dataset, load_from_disk
from loguru import logger


def load_dataset_from_dataset_id(
    dataset_id: str,
    language_id: str | None = None,
    feature_extractor: WhisperFeatureExtractor | None = None,
    tokenizer: WhisperTokenizer | None = None,
    local_train_split: float = 0.8,
) -> DatasetDict:
    """
    This function attempts to load a dataset based on certain scenarios:
    1. The dataset_id is a local path to an already processed dataset directory.
        We make sure the final folder name of that path is the PROC_DATASET_DIR constant.

    2. The dataset_id is a path to a local dataset directory.
        If that directory contains a PROC_DATASET_DIR folder, then we load the processed version directly.
        If not, we load the local dataset, process it, save it locally and return it.

    3. The dataset_id is an HF dataset repo id.
        We first check if that dataset had already been processed and saved locally under the artifacts directory
        If not, we load it, process it, save it locally under artifacts and return it.

    Args:
        dataset_id: Path to a processed dataset directory or local dataset directory or HuggingFace dataset ID.
        language_id (Only used for the HF dataset case): Language identifier for the dataset (e.g., 'en' for English)
        feature_extractor (Only used for non-processed datasets): Whisper feature extractor for processing audio inputs
        tokenizer: (Only used for non-processed datasets) Whisper tokenizer for processing text inputs
        local_train_split: (Only used for non-processed, local, custom datasets) Percentage split of train/test sets

    Returns:
        DatasetDict: A processed dataset ready for training with train/test splits

    Raises:
        ValueError: If the dataset cannot be found locally or on HuggingFace
    """
    dataset_path = Path(dataset_id)
    proc_dataset_path = dataset_path.resolve() / PROC_DATASET_DIR

    # Check if dataset_id is a local directory
    if dataset_path.is_dir():
        # Check if the local dataset already contains a processed version or is itself already the processed version
        if proc_dataset_path.is_dir() or Path(dataset_path.name) == PROC_DATASET_DIR:
            logger.info(
                f"Found processed dataset at {dataset_id}. Loading it directly and skipping processing."
            )
            return load_from_disk(dataset_id)

        logger.info(f"Found local dataset at {dataset_id}.")
        dataset = _load_local_dataset(dataset_id, train_split=local_train_split)
    elif repo_exists(dataset_id, repo_type="dataset"):
        # If it's an HF dataset id, check if we had already saved a processed version under the artifacts directory
        proc_dataset_path = f"./artifacts/{language_id}_{dataset_id.replace('/', '_')}/{PROC_DATASET_DIR}"
        if Path(proc_dataset_path).is_dir():
            logger.info(
                f"Found processed version of {dataset_id} at {proc_dataset_path}. Loading it directly and skipping processing."
            )
            return load_from_disk(dataset_id)
        logger.info(f"Loading HuggingFace dataset from {dataset_id}.")
        dataset = _load_common_voice(dataset_id, language_id)
    else:
        raise ValueError(
            f"Could not find dataset {dataset_id}, neither locally nor at HuggingFace. "
            f"If its a private repo, make sure you are logged in locally."
        )

    logger.info("Processing dataset...")
    dataset = _process_dataset(dataset, feature_extractor, tokenizer, proc_dataset_path)
    logger.info(
        f"Processed dataset saved at {proc_dataset_path}. Future runs of {dataset_id} will automatically use "
        f"this processed version."
    )
    return dataset


def _load_common_voice(dataset_id: str, language_id: str) -> DatasetDict:
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


def _load_local_dataset(dataset_dir: str, train_split: float = 0.8) -> DatasetDict:
    """
    Load sentences and accompanied recorded audio files into a pandas DataFrame, then split into train/test and finally
    load it into two distinct train Dataset and test Dataset.

    Sentences and audio files should be indexed like this: <index>: <sentence> should be accompanied by rec_<index>.wav

    Args:
        dataset_dir (str): path to the local dataset, expecting a text.csv and .wav files under the directory
        train_split (float): percentage split of the dataset to train+validation and test set

    Returns:
        DatasetDict: HF Dataset dictionary in the same exact format as the Common Voice dataset from load_common_voice
    """
    text_file = dataset_dir + "/text.csv"

    dataframe = pd.read_csv(text_file)
    audio_files = sorted(
        [f"{dataset_dir}/{f}" for f in os.listdir(dataset_dir) if f.endswith(".wav")]
    )

    dataframe["audio"] = audio_files
    train_index = round(len(dataframe) * train_split)

    my_data = DatasetDict()
    my_data["train"] = Dataset.from_pandas(dataframe[:train_index])
    my_data["test"] = Dataset.from_pandas(dataframe[train_index:])

    return my_data


def _process_dataset(
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
