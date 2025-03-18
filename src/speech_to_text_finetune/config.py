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

    Attributes:
        model_id: HF model id of a Whisper model used for finetuning
        dataset_id: HF dataset id of a Common Voice dataset version, ideally from the mozilla-foundation repo
        language: registered language string that is supported by the Common Voice dataset
        repo_name: used both for local dir and HF, "default" will create a name based on the model and language id
        n_train_samples: explicitly set how many samples to train+validate on. If -1, use all train+val data available
        n_test_samples: explicitly set how many samples to evaluate on. If -1, use all eval data available
        training_hp: store selective hyperparameter values from Seq2SeqTrainingArguments
    """

    model_id: str
    dataset_id: str
    language: str
    repo_name: str
    n_train_samples: int
    n_test_samples: int
    training_hp: TrainingConfig


PROC_DATASET_DIR = "processed_version"
