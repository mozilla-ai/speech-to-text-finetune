from typing import Dict

from evaluate import EvaluationModule
from huggingface_hub import (
    ModelCard,
    HfApi,
    ModelCardData,
    EvalResult,
)
from transformers import EvalPrediction, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def compute_wer_cer_metrics(
    pred: EvalPrediction,
    processor: WhisperProcessor,
    wer: EvaluationModule,
    cer: EvaluationModule,
    normalizer: BasicTextNormalizer,
) -> Dict:
    """
    Word Error Rate (wer) is a metric that measures the ratio of errors the ASR model makes given a transcript to the
    total words spoken. Lower is better.
    Character Error Rate (cer) is similar to wer, but operates on character instead of word. This metric is better
    suited for languages with no concept of "word" like Chinese or Japanese. Lower is better.

    More info: https://huggingface.co/learn/audio-course/en/chapter5/fine-tuning#evaluation-metrics

    Note 1: WER/CER can be larger than 1.0, if the number of insertions I is larger than the number of correct words C.
    Note 2: WER/CER doesn't tell the whole story and is not fully representative of the quality of the ASR model.

    Args:
        pred (EvalPrediction): Transformers object that holds predicted tokens and ground truth labels
        processor (WhisperProcessor): Whisper processor used to decode tokens to strings
        wer (EvaluationModule): module that calls the computing function for WER
        cer (EvaluationModule): module that calls the computing function for CER
        normalizer (BasicTextNormalizer): Normalizer from Whisper
    Returns:
        wer (Dict): computed WER metric
    """

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * wer.compute(predictions=pred_str, references=label_str)
    cer_ortho = 100 * cer.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i]
        for i in range(len(pred_str_norm))
        if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * wer.compute(predictions=pred_str_norm, references=label_str_norm)
    cer = 100 * cer.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer, "cer_ortho": cer_ortho, "cer": cer}


def get_hf_username() -> str:
    return HfApi().whoami()["name"]


def create_model_card(
    model_id: str,
    dataset_id: str,
    language_id: str,
    language: str,
    n_train_samples: int,
    n_eval_samples: int,
    baseline_eval_results: Dict,
    ft_eval_results: Dict,
) -> ModelCard:
    """
    Create and upload a custom Model Card (https://huggingface.co/docs/hub/model-cards) to the Hugging Face repo
    of the finetuned model that highlights the evaluation results before and after finetuning.
    """
    card_metadata = ModelCardData(
        model_name=f"Finetuned {model_id} on {language}",
        base_model=model_id,
        datasets=[dataset_id.split("/")[-1]],
        language=language_id,
        license="apache-2.0",
        library_name="transformers",
        eval_results=[
            EvalResult(
                task_type="automatic-speech-recognition",
                task_name="Speech-to-Text",
                dataset_type="common_voice",
                dataset_name=f"Common Voice ({language})",
                metric_type="wer",
                metric_value=round(ft_eval_results["eval_wer"], 3),
            )
        ],
    )
    content = f"""
---
{card_metadata.to_yaml()}
---

# Finetuned {model_id} on {n_train_samples} {language} training audio samples from {dataset_id}.

This model was created from the Mozilla.ai Blueprint:
[speech-to-text-finetune](https://github.com/mozilla-ai/speech-to-text-finetune).

## Evaluation results on {n_eval_samples} audio samples of {language}:

### Baseline model (before finetuning) on {language}
- Word Error Rate (Normalized): {round(baseline_eval_results["eval_wer"], 3)}
- Word Error Rate (Orthographic): {round(baseline_eval_results["eval_wer_ortho"], 3)}
- Character Error Rate (Normalized): {round(baseline_eval_results["eval_cer"], 3)}
- Character Error Rate (Orthographic): {round(baseline_eval_results["eval_cer_ortho"], 3)}
- Loss: {round(baseline_eval_results["eval_loss"], 3)}

### Finetuned model (after finetuning) on {language}
- Word Error Rate (Normalized): {round(ft_eval_results["eval_wer"], 3)}
- Word Error Rate (Orthographic): {round(ft_eval_results["eval_wer_ortho"], 3)}
- Character Error Rate (Normalized): {round(ft_eval_results["eval_cer"], 3)}
- Character Error Rate (Orthographic): {round(ft_eval_results["eval_cer_ortho"], 3)}
- Loss: {round(ft_eval_results["eval_loss"], 3)}
"""

    return ModelCard(content)


def update_hf_model_card_with_fleurs_results(
    model_repo_id: str,
    language: str,
    ft_eval_results: Dict,
) -> None:
    """
    Update the HF Model Card with the evaluation results from the FLEURS dataset.
    """
    model_card = ModelCard.load(model_repo_id)
    model_card.content += f"""
### Finetuned model (after finetuning) on the {language} FLEURS test set (total of {ft_eval_results["n_eval_samples"]} samples)
- Word Error Rate (Normalized): {round(ft_eval_results["eval_wer"], 3)}
- Word Error Rate (Orthographic): {round(ft_eval_results["eval_wer_ortho"], 3)}
- Character Error Rate (Normalized): {round(ft_eval_results["eval_cer"], 3)}
- Character Error Rate (Orthographic): {round(ft_eval_results["eval_cer_ortho"], 3)}
- Loss: {round(ft_eval_results["eval_loss"], 3)}
"""
    model_card.push_to_hub(model_repo_id)
