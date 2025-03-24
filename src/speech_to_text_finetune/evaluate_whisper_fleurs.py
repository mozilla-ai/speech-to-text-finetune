from datasets import load_dataset
from functools import partial

from transformers import (
    Seq2SeqTrainer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import torch
import evaluate
from loguru import logger

from speech_to_text_finetune.data_process import (
    load_subset_of_dataset,
    process_dataset,
    DataCollatorSpeechSeq2SeqWithPadding,
)
from speech_to_text_finetune.utils import compute_wer_cer_metrics


def evaluate_fleurs(
    model_id: str,
    lang_code: str,
    language: str,
    eval_batch_size: int,
    n_eval_samples: int,
    fp16: bool,
):
    logger.info(f"Loading Fleurs dataset for language: {language}")

    dataset = load_dataset(
        "google/fleurs", lang_code, trust_remote_code=True, split="test"
    )

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info(f"Loading {model_id} on {device} and configuring it for {language}.")
    # TODO: Load processed dataset if it exists!!
    processor = WhisperProcessor.from_pretrained(
        model_id, language=language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    # set language and task for generation during inference and re-enable cache
    model.generate = partial(
        model.generate, language=language.lower(), task="transcribe", use_cache=True
    )

    dataset = load_subset_of_dataset(dataset, n_samples=n_eval_samples)
    dataset = process_dataset(
        dataset=dataset,
        processor=processor,
        proc_dataset_path=f"fleurs_{lang_code}_{n_eval_samples}_samples",
    )

    wer = evaluate.load("wer")
    cer = evaluate.load("cer")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = Seq2SeqTrainer(
        args=Seq2SeqTrainingArguments(
            fp16=fp16,
            per_device_eval_batch_size=eval_batch_size,
            predict_with_generate=True,
            generation_max_length=225,
        ),
        model=model,
        eval_dataset=dataset,
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

    logger.info(f"Start evaluation on {dataset.num_rows} audio samples.")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation complete. Results:\n\t {eval_results}")
    return eval_results


if __name__ == "__main__":
    evaluate_fleurs(
        model_id="openai/whisper-tiny",
        lang_code="sw_ke",
        language="Swahili",
        eval_batch_size=8,
        n_eval_samples=-1,
        fp16=True,
    )
