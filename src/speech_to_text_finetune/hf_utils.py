from typing import Dict
from huggingface_hub import (
    ModelCard,
    HfApi,
    ModelCardData,
    EvalResult,
)


def get_hf_username() -> str:
    return HfApi().whoami()["name"]


def upload_custom_hf_model_card(
    hf_repo_name: str,
    model_id: str,
    dataset_id: str,
    language_id: str,
    language: str,
    n_train_samples: int,
    n_eval_samples: int,
    baseline_eval_results: Dict,
    ft_eval_results: Dict,
) -> None:
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

    card = ModelCard(content)
    card.push_to_hub(hf_repo_name)
