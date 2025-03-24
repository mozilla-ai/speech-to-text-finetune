import json
from functools import partial

from transformers import (
    Seq2SeqTrainer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
import torch
from typing import Dict, Tuple
import evaluate
from loguru import logger

from speech_to_text_finetune.config import load_config
from speech_to_text_finetune.data_process import (
    DataCollatorSpeechSeq2SeqWithPadding,
    load_dataset_from_dataset_id,
    try_find_processed_version,
    process_dataset,
    load_subset_of_dataset,
)
from speech_to_text_finetune.utils import (
    get_hf_username,
    upload_custom_hf_model_card,
    compute_wer_cer_metrics,
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

    language_id = TO_LANGUAGE_CODE.get(cfg.language.lower())
    if not language_id:
        raise ValueError(
            f"\nThis language is not inherently supported by this Whisper model. However you can still “teach” Whisper "
            f"the language of your choice!\nVisit https://glottolog.org/, find which language is most closely "
            f"related to {cfg.language} from the list of supported languages below, and update your config file with "
            f"that language.\n{json.dumps(TO_LANGUAGE_CODE, indent=4, sort_keys=True)}."
        )

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
    processor = WhisperProcessor.from_pretrained(
        cfg.model_id, language=cfg.language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model_id)

    # disable cache during training since it's incompatible with gradient checkpointing
    model.config.use_cache = False
    # set language and task for generation during inference and re-enable cache
    model.generate = partial(
        model.generate, language=cfg.language.lower(), task="transcribe", use_cache=True
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=local_output_dir,
        hub_model_id=hf_repo_name,
        report_to=["tensorboard"],
        **cfg.training_hp.model_dump(),
    )

    if proc_dataset := try_find_processed_version(
        dataset_id=cfg.dataset_id, language_id=language_id
    ):
        logger.info(
            f"Loading processed dataset version of {cfg.dataset_id} and skipping processing."
        )
        dataset = proc_dataset
        dataset["train"] = load_subset_of_dataset(dataset["train"], cfg.n_train_samples)
        dataset["test"] = load_subset_of_dataset(dataset["test"], cfg.n_test_samples)
    else:
        logger.info(f"Loading {cfg.dataset_id}. Language selected {cfg.language}")
        dataset, save_proc_dataset_dir = load_dataset_from_dataset_id(
            dataset_id=cfg.dataset_id,
            language_id=language_id,
        )
        dataset["train"] = load_subset_of_dataset(dataset["train"], cfg.n_train_samples)
        dataset["test"] = load_subset_of_dataset(dataset["test"], cfg.n_test_samples)
        logger.info("Processing dataset...")
        dataset = process_dataset(
            dataset=dataset,
            processor=processor,
            batch_size=cfg.training_hp.per_device_train_batch_size,
            proc_dataset_path=save_proc_dataset_dir,
        )
        logger.info(
            f"Processed dataset saved at {save_proc_dataset_dir}. Future runs of {cfg.dataset_id} will "
            f"automatically use this processed version."
        )

    wer = evaluate.load("wer")
    cer = evaluate.load("cer")

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=partial(
            compute_wer_cer_metrics,
            processor=processor,
            wer=wer,
            cer=cer,
            normalizer=BasicTextNormalizer(),
        ),
        processing_class=processor.feature_extractor,
    )

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


if __name__ == "__main__":
    run_finetuning(config_path="example_data/config.yaml")
