import yaml
from pydantic import BaseModel


def load_config(config_path: str):
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    return Config(**config_dict)


class TrainingConfig(BaseModel):
    """
    More info at https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments
    """

    push_to_hub: bool
    hub_private_repo: bool
    max_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    gradient_checkpointing: bool
    fp16: bool
    eval_strategy: str
    per_device_eval_batch_size: int
    predict_with_generate: bool
    generation_max_length: int
    save_steps: int
    logging_steps: int
    load_best_model_at_end: bool
    save_total_limit: int
    metric_for_best_model: str
    greater_is_better: bool


class Config(BaseModel):
    """
    Store configuration used for finetuning

    Args:
        model_id (str): HF model id of a Whisper model used for finetuning
        dataset_id (str): HF dataset id of a Common Voice dataset version, ideally from the mozilla-foundation repo
        dataset_source (str): can be "HF" or "local", to determine from where to fetch the dataset
        language (str): registered language string that is supported by the Common Voice dataset
        repo_name (str): used both for local dir and HF, "default" will create a name based on the model and language id
        training_hp (TrainingConfig): store selective hyperparameter values from Seq2SeqTrainingArguments
    """

    model_id: str
    dataset_id: str
    dataset_source: str
    language: str
    repo_name: str
    training_hp: TrainingConfig
