from functools import partial

from transformers import (
    Seq2SeqTrainer,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    EvalPrediction,
)
import torch
from datasets import Audio
from typing import Dict, Tuple
from language_registry import common_voice_languages
import evaluate
from evaluate import EvaluationModule
from loguru import logger
from data import (
    load_common_voice,
    prepare_dataset_for_whisper,
    DataCollatorSpeechSeq2SeqWithPadding,
)
from hf_utils import (
    get_hf_username,
    upload_custom_hf_model_card,
)

hf_username = get_hf_username()
dataset_id_cv = "mozilla-foundation/common_voice_17_0"
model_id_whisper = "openai/whisper-tiny"
test_language = "Greek"

test_repo_name = "testing"  # None for default name, or set your own
push_to_hf = True
make_repo_private = False


def run_finetuning(
    model_id: str, dataset_id: str, language: str, repo_name: str | None
) -> Tuple[Dict, Dict]:
    """
    Complete pipeline for preprocessing the Common Voice dataset and then finetuning a Whisper model on it.

    Args:
        model_id (str): HF model id of a Whisper model used for finetuning
        dataset_id (str): HF dataset id of a Common Voice dataset version, ideally from the mozilla-foundation repo
        language (str): registered language string that is supported by the Common Voice dataset
        repo_name (str): repo ID that will be used for storing artifacts both locally and on HF

    Returns:
        Tuple[Dict, Dict]: evaluation metrics from the baseline and the finetuned models
    """

    language_id = common_voice_languages[language]

    if not repo_name:
        repo_name = f"{model_id.split('/')[1]}-{language_id}"
    hf_repo_name = f"{hf_username}/{repo_name}"
    local_output_dir = f"./artifacts/{repo_name}"

    logger.info(
        f"Finetuning job will soon start. "
        f"Results will be saved local at {local_output_dir} uploaded in HF at {hf_repo_name}. "
        f"Private repo is set to {make_repo_private}."
    )

    logger.info(f"Loading the {language} subset from the {dataset_id} dataset.")
    dataset = load_common_voice(dataset_id, language_id)

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    logger.info(f"Loading {model_id} on {device} and configuring it for {language}.")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_id, language=language, task="transcribe"
    )
    processor = WhisperProcessor.from_pretrained(
        model_id, language=language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_id)

    model.generation_config.language = language.lower()
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    logger.info("Preparing dataset...")
    # Create a new column that consists of the resampled audio samples in the right sample rate for whisper
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(
        prepare_dataset_for_whisper,
        fn_kwargs={"feature_extractor": feature_extractor, "tokenizer": tokenizer},
        remove_columns=dataset.column_names["train"],
        num_proc=2,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=50,
        max_steps=1000,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=250,
        eval_steps=250,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
        push_to_hub=push_to_hf,
        hub_model_id=hf_repo_name,
        hub_private_repo=make_repo_private,
    )

    metric = evaluate.load("wer")

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=partial(
            compute_word_error_rate, tokenizer=tokenizer, metric=metric
        ),
        processing_class=processor.feature_extractor,
    )

    processor.save_pretrained(training_args.output_dir)

    logger.info(
        f"Before finetuning, run evaluation on the baseline model {model_id} to easily compare performance"
        f" before and after finetuning"
    )
    baseline_eval_results = trainer.evaluate()
    logger.info(f"Baseline evaluation complete. Results:\n\t {baseline_eval_results}")

    logger.info(
        f"Start finetuning job on {dataset['train'].num_rows} audio samples. Monitor training metrics in real time in "
        f"a local tensorboard server by running in a new terminal: tensorboard --logdir {training_args.output_dir}/runs"
    )
    trainer.train()
    logger.info("Finetuning job complete.")

    logger.info(f"Start evaluation on {dataset['test'].num_rows} audio samples.")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation complete. Results:\n\t {eval_results}")

    if push_to_hf:
        logger.info(f"Uploading model and eval results to HuggingFace: {hf_repo_name}")
        trainer.push_to_hub()
        upload_custom_hf_model_card(
            hf_repo_name=hf_repo_name,
            model_id=model_id,
            dataset_id=dataset_id,
            language_id=language_id,
            language=language,
            n_train_samples=dataset["train"].num_rows,
            n_eval_samples=dataset["test"].num_rows,
            baseline_eval_results=baseline_eval_results,
            ft_eval_results=eval_results,
        )

    return baseline_eval_results, eval_results


def compute_word_error_rate(
    pred: EvalPrediction, tokenizer: WhisperTokenizer, metric: EvaluationModule
) -> Dict:
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


if __name__ == "__main__":
    run_finetuning(
        model_id=model_id_whisper,
        dataset_id=dataset_id_cv,
        language=test_language,
        repo_name=test_repo_name,
    )
