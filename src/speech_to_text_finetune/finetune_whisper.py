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
from typing import Dict, Tuple
import evaluate
from evaluate import EvaluationModule
from loguru import logger

from speech_to_text_finetune.config import load_config, LANGUAGES_NAME_TO_ID
from speech_to_text_finetune.data_process import (
    DataCollatorSpeechSeq2SeqWithPadding,
    load_dataset_from_dataset_id,
    try_find_processed_version,
    process_dataset,
)
from speech_to_text_finetune.hf_utils import (
    get_hf_username,
    upload_custom_hf_model_card,
)


def run_finetuning(
    config_path: str = "config.yaml",
) -> Tuple[Dict, Dict]:
    """
    Complete pipeline for preprocessing the Common Voice dataset and then finetuning a Whisper model on it.

    Args:
        config_path (str): yaml filepath that follows the format defined in config.py

    Returns:
        Tuple[Dict, Dict]: evaluation metrics from the baseline and the finetuned models
    """
    cfg = load_config(config_path)

    language_id = LANGUAGES_NAME_TO_ID[cfg.language]

    if cfg.repo_name == "default":
        cfg.repo_name = f"{cfg.model_id.split('/')[1]}-{language_id}"
    local_output_dir = f"./artifacts/{cfg.repo_name}"

    logger.info(f"Finetuning starts soon, results saved locally at {local_output_dir}")
    hf_repo_name = ""
    if cfg.training_hp.push_to_hub:
        hf_username = get_hf_username()
        hf_repo_name = f"{hf_username}/{cfg.repo_name}"
        logger.info(
            f"Results will also be uploaded in HF at {hf_repo_name}. "
            f"Private repo is set to {cfg.training_hp.hub_private_repo}."
        )

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info(
        f"Loading {cfg.model_id} on {device} and configuring it for {cfg.language}."
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.model_id)
    tokenizer = WhisperTokenizer.from_pretrained(
        cfg.model_id, language=cfg.language, task="transcribe"
    )
    processor = WhisperProcessor.from_pretrained(
        cfg.model_id, language=cfg.language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model_id)

    model.generation_config.language = cfg.language.lower()
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=local_output_dir,
        hub_model_id=hf_repo_name,
        report_to=["tensorboard"],
        **cfg.training_hp.model_dump(),
    )

    metric = evaluate.load("wer")

    if proc_dataset := try_find_processed_version(
        dataset_id=cfg.dataset_id, language_id=language_id
    ):
        logger.info(
            f"Loading processed dataset version of {cfg.dataset_id} and skipping processing."
        )
        dataset = proc_dataset
    else:
        logger.info(f"Loading {cfg.dataset_id}. Language selected {cfg.language}")
        dataset, save_proc_dataset_dir = load_dataset_from_dataset_id(
            dataset_id=cfg.dataset_id,
            language_id=language_id,
            local_train_split=0.8,
        )
        logger.info("Processing dataset...")
        dataset = process_dataset(
            dataset=dataset,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            proc_dataset_path=save_proc_dataset_dir,
        )
        logger.info(
            f"Processed dataset saved at {save_proc_dataset_dir}. Future runs of {cfg.dataset_id} will "
            f"automatically use this processed version."
        )

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

    feature_extractor.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    logger.info(
        f"Before finetuning, run evaluation on the baseline model {cfg.model_id} to easily compare performance"
        f" before and after finetuning"
    )
    baseline_eval_results = trainer.evaluate()
    logger.info(f"Baseline evaluation complete. Results:\n\t {baseline_eval_results}")

    logger.info(
        f"Start finetuning job on {dataset['train'].num_rows} audio samples. Monitor training metrics in real time in "
        f"a local tensorboard server by running in a new terminal: tensorboard --logdir {training_args.output_dir}/runs"
    )
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Stopping the finetuning job prematurely...")
    else:
        logger.info("Finetuning job complete.")

    logger.info(f"Start evaluation on {dataset['test'].num_rows} audio samples.")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation complete. Results:\n\t {eval_results}")

    if cfg.training_hp.push_to_hub:
        logger.info(f"Uploading model and eval results to HuggingFace: {hf_repo_name}")
        trainer.push_to_hub()
        upload_custom_hf_model_card(
            hf_repo_name=hf_repo_name,
            model_id=cfg.model_id,
            dataset_id=cfg.dataset_id,
            language_id=language_id,
            language=cfg.language,
            n_train_samples=dataset["train"].num_rows,
            n_eval_samples=dataset["test"].num_rows,
            baseline_eval_results=baseline_eval_results,
            ft_eval_results=eval_results,
        )

    logger.info(f"Find your final, best performing model at {local_output_dir}")
    return baseline_eval_results, eval_results


def compute_word_error_rate(
    pred: EvalPrediction, tokenizer: WhisperTokenizer, metric: EvaluationModule
) -> Dict:
    """
    Word Error Rate (wer) is a metric that measures the ratio of errors the ASR model makes given a transcript to the
    total words spoken. Lower is better.
    To identify an "error" we measure the difference between the ASR generated transcript and the
    ground truth transcript using the following formula:
    - S is the number of substitutions (number of words ASR swapped for different words from the ground truth)
    - D is the number of deletions (number of words ASR skipped / didn't generate compared to the ground truth)
    - I is the number of insertions (number of additional words ASR generated, not found in the ground truth)
    - C is the number of correct words (number of words that are identical between ASR and ground truth scripts)

    then: WER = (S+D+I) / (S+D+C)

    Note 1: WER can be larger than 1.0, if the number of insertions I is larger than the number of correct words C.
    Note 2: WER doesn't tell the whole story and is not fully representative of the quality of the ASR model.

    Args:
        pred (EvalPrediction): Transformers object that holds predicted tokens and ground truth labels
        tokenizer (WhisperTokenizer): Whisper tokenizer used to decode tokens to strings
        metric (EvaluationModule): module that calls the computing function for WER
    Returns:
        wer (Dict): computed WER metric
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


if __name__ == "__main__":
    run_finetuning(config_path="example_data/config.yaml")
