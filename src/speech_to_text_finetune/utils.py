from typing import Dict

from evaluate import EvaluationModule
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
