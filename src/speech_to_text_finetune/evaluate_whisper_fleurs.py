import argparse
from functools import partial
from typing import Dict

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
    DataCollatorSpeechSeq2SeqWithPadding,
    load_and_proc_hf_fleurs,
)
from speech_to_text_finetune.utils import (
    compute_wer_cer_metrics,
    update_hf_model_card_with_fleurs_results,
)


def evaluate_fleurs(
    model_id: str,
    lang_code: str,
    language: str,
    eval_batch_size: int,
    n_test_samples: int,
    fp16: bool,
) -> Dict:
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info(f"Loading {model_id} on {device} and configuring it for {language}.")
    if fp16 and not torch.cuda.is_available():
        logger.warning("FP16 is enabled but no GPU is available. Disabling FP16.")
        fp16 = False

    processor = WhisperProcessor.from_pretrained(
        model_id, language=language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    # set language and task for generation during inference and re-enable cache
    model.generate = partial(
        model.generate,
        language=language.lower(),
        task="transcribe",
        use_cache=True,
    )

    logger.info(f"Loading Fleurs dataset for language: {language}")
    dataset = load_and_proc_hf_fleurs(
        language_id=lang_code,
        n_test_samples=n_test_samples,
        processor=processor,
        eval_batch_size=eval_batch_size,
    )

    wer = evaluate.load("wer")
    cer = evaluate.load("cer")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = Seq2SeqTrainer(
        args=Seq2SeqTrainingArguments(
            fp16=fp16,
            per_device_eval_batch_size=eval_batch_size,
            predict_with_generate=True,
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
    eval_results["n_eval_samples"] = len(dataset)
    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper model on Fleurs dataset"
    )
    parser.add_argument(
        "--model_id", type=str, default="openai/whisper-tiny", help="Model identifier"
    )
    parser.add_argument(
        "--lang_code", type=str, default="sw_ke", help="Language code for the dataset"
    )
    parser.add_argument(
        "--language", type=str, default="Swahili", help="Language for transcription"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--n_test_samples",
        type=int,
        default=-1,
        help="Number of test samples to use (-1 for all)",
    )
    parser.add_argument(
        "--fp16", type=bool, default=True, help="Enable FP16 precision if GPU available"
    )
    parser.add_argument(
        "--update_hf_repo",
        type=bool,
        default=False,
        help="Update the HF Model Card repo with the FLEURS evaluation results",
    )

    args = parser.parse_args()

    results = evaluate_fleurs(
        model_id=args.model_id,
        lang_code=args.lang_code,
        language=args.language,
        eval_batch_size=args.eval_batch_size,
        n_test_samples=args.n_test_samples,
        fp16=args.fp16,
    )

    if args.update_hf_repo:
        update_hf_model_card_with_fleurs_results(
            args.model_id, language=args.language, ft_eval_results=results
        )
