from transformers import (
    Seq2SeqTrainer,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig,
)
import torch
from typing import Dict, Tuple
from loguru import logger

from speech_to_text_finetune.config import load_config, LANGUAGES_NAME_TO_ID
from speech_to_text_finetune.data_process import (
    load_common_voice,
    load_local_dataset,
    DataCollatorSpeechSeq2SeqWithPadding,
    process_dataset,
)
from speech_to_text_finetune.hf_utils import (
    get_hf_username,
    upload_custom_hf_model_card,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from speech_to_text_finetune.peft_utils import SavePeftModelCallback, evaluate_w_peft


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

    logger.info(f"Loading the {cfg.language} subset from the {cfg.dataset_id} dataset.")
    if cfg.dataset_source == "HF":
        dataset = load_common_voice(cfg.dataset_id, language_id)
    elif cfg.dataset_source == "local":
        dataset = load_local_dataset(cfg.dataset_id, train_split=0.8)
    else:
        raise ValueError(f"Unknown dataset source {cfg.dataset_source}")

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

    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.model_id,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=cfg.peft_hp.load_in_8bit),
    )

    model = prepare_model_for_kbit_training(model)

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.peft_hp.rank,
            lora_alpha=cfg.peft_hp.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=cfg.peft_hp.lora_dropout,
            bias=cfg.peft_hp.bias,
        ),
    )
    model.print_trainable_parameters()

    logger.info("Preparing dataset...")
    dataset = process_dataset(dataset, feature_extractor, tokenizer)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, bos_token_id=tokenizer.bos_token_id
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=local_output_dir,
        hub_model_id=hf_repo_name,
        report_to=["tensorboard"],
        **cfg.training_hp.model_dump(),
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
        callbacks=[SavePeftModelCallback],
    )

    # model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    processor.save_pretrained(training_args.output_dir)

    logger.info(
        f"Start finetuning job on {dataset['train'].num_rows} audio samples. Monitor training metrics in real time in "
        f"a local tensorboard server by running in a new terminal: tensorboard --logdir {training_args.output_dir}/runs"
    )
    trainer.train()
    logger.info("Finetuning job complete.")

    logger.info(f"Start evaluation on {dataset['test'].num_rows} audio samples.")
    eval_results = evaluate_w_peft(
        model=model,
        test_dataset=dataset["test"],
        data_collator=data_collator,
        processor=processor,
        language=cfg.language,
    )
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
            baseline_eval_results={},
            ft_eval_results=eval_results,
        )

    return eval_results


if __name__ == "__main__":
    run_finetuning(config_path="example_data/config.yaml")
