import importlib.util
from typing import Dict
from huggingface_hub import ModelCard, HfApi, ModelCardData, hf_hub_download


def get_hf_username() -> str:
    return HfApi().whoami()["name"]


def get_available_languages_in_cv(dataset_id: str) -> Dict:
    """
    Downloads a languages.py file from a Common Voice dataset repo which stores all languages available.
    Then, dynamically imports the file as a module and returns the dictionary defined inside.
    Since the dictionary is in the format {<ISO-639-id>: <Full language name>} , e.g. {'ab': 'Abkhaz'}
    We also swap to use the full language name as key and the ISO id as value instead.

    Args:
        dataset_id: It needs to be a specific Common Voice dataset id, e.g. mozilla-foundation/common_voice_17_0

    Returns:
        Dict: A language mapping dictionary in the format {<Full language name>: <ISO-639-id>} , e.g. {'Abkhaz': 'ab'}
    """
    filepath = hf_hub_download(
        repo_id=dataset_id, filename="languages.py", repo_type="dataset", local_dir="."
    )

    spec = importlib.util.spec_from_file_location("languages_map_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    lang_id_to_name = module.LANGUAGES
    lang_name_to_id = dict((v, k) for k, v in lang_id_to_name.items())
    return lang_name_to_id


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
