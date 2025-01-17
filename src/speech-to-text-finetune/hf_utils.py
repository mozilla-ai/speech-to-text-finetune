from typing import Dict
from huggingface_hub import ModelCard, HfApi, ModelCardData


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
    card_metadata = ModelCardData(
        base_model=model_id,
        datasets=[dataset_id],
        language=language_id,
        metrics=["wer"],
    )
    content = f"""
    ---
    {card_metadata.to_yaml()}
    ---

    # Finetuned version of {model_id} on {n_train_samples} {language} training audio samples from {dataset_id}.

    This model was created from the Mozilla.ai Blueprint:
    [speech-to-text-finetune](https://github.com/mozilla-ai/speech-to-text-finetune).

    ## Evaluation results on {n_eval_samples} audio samples of {language}

    ### Baseline model (before finetuning) on {language}
    - Word Error Rate: {round(baseline_eval_results["eval_wer"], 3)}
    - Loss: {round(baseline_eval_results["eval_loss"], 3)}

    ### Finetuned model (after finetuning) on {language}
    - Word Error Rate: {round(ft_eval_results["eval_wer"], 3)}
    - Loss: {round(ft_eval_results["eval_loss"], 3)}
    """

    card = ModelCard(content)
    card.push_to_hub(hf_repo_name)
