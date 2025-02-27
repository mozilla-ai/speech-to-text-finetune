import os
import tempfile
import yaml
import pytest

from speech_to_text_finetune.config import load_config, Config, TrainingConfig, LANGUAGES_NAME_TO_ID
from pydantic import ValidationError
def create_temp_config_file(config_data, tmp_path):
    """Helper function to create a temporary YAML config file."""
    file_path = tmp_path / "config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config_data, f)
    return file_path

def test_load_config_valid(tmp_path):
    """Test load_config returns a Config object with proper fields from a valid YAML file."""
    config_data = {
        "model_id": "openai/whisper-small",
        "dataset_id": "mozilla-foundation/common_voice_en",
        "dataset_source": "HF",
        "language": "English",
        "repo_name": "whisper_en",
        "n_train_samples": 100,
        "n_test_samples": 50,
        "training_hp": {
            "push_to_hub": False,
            "hub_private_repo": True,
            "max_steps": 1000,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "learning_rate": 5e-5,
            "warmup_steps": 500,
            "gradient_checkpointing": True,
            "fp16": True,
            "eval_strategy": "steps",
            "per_device_eval_batch_size": 8,
            "predict_with_generate": False,
            "generation_max_length": 128,
            "save_steps": 100,
            "logging_steps": 50,
            "load_best_model_at_end": True,
            "save_total_limit": 3,
            "metric_for_best_model": "wer",
            "greater_is_better": False
        }
    }
    file_path = create_temp_config_file(config_data, tmp_path)
    config_obj = load_config(str(file_path))
    # Verify that the config_obj is an instance of Config and has a TrainingConfig inside
    assert isinstance(config_obj, Config)
    assert isinstance(config_obj.training_hp, TrainingConfig)
    assert config_obj.model_id == "openai/whisper-small"
    assert config_obj.training_hp.learning_rate == 5e-5

def test_training_config_validation():
    """Test TrainingConfig validation raises error when required fields are missing."""
    valid_data = {
        "push_to_hub": True,
        "hub_private_repo": False,
        "max_steps": 2000,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "learning_rate": 3e-4,
        "warmup_steps": 100,
        "gradient_checkpointing": False,
        "fp16": False,
        "eval_strategy": "no",
        "per_device_eval_batch_size": 4,
        "predict_with_generate": True,
        "generation_max_length": 150,
        "save_steps": 500,
        "logging_steps": 100,
        "load_best_model_at_end": False,
        "save_total_limit": 2,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True
    }
    # Removing one required field to trigger validation error
    invalid_data = valid_data.copy()
    invalid_data.pop("learning_rate")

    with pytest.raises(Exception):
        TrainingConfig(**invalid_data)

def test_languages_mapping():
    """Test that LANGUAGES_NAME_TO_ID returns correct mappings."""
    assert LANGUAGES_NAME_TO_ID["English"] == "en"
    assert LANGUAGES_NAME_TO_ID.get("NonExistent") is None
def test_load_config_empty_file(tmp_path):
    """Test that load_config raises an error when the YAML file is empty."""
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("")
    with pytest.raises(TypeError):
        # When yaml.safe_load returns None, passing None to Config(**...) causes a TypeError
        load_config(str(empty_file))

def test_load_config_missing_field(tmp_path):
    """Test that load_config raises a validation error when a required field is missing."""
    config_data = {
        # "model_id" is missing,
        "dataset_id": "mozilla-foundation/common_voice_en",
        "dataset_source": "HF",
        "language": "English",
        "repo_name": "whisper_en",
        "n_train_samples": 100,
        "n_test_samples": 50,
        "training_hp": {
            "push_to_hub": False,
            "hub_private_repo": True,
            "max_steps": 1000,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "learning_rate": 5e-5,
            "warmup_steps": 500,
            "gradient_checkpointing": True,
            "fp16": True,
            "eval_strategy": "steps",
            "per_device_eval_batch_size": 8,
            "predict_with_generate": False,
            "generation_max_length": 128,
            "save_steps": 100,
            "logging_steps": 50,
            "load_best_model_at_end": True,
            "save_total_limit": 3,
            "metric_for_best_model": "wer",
            "greater_is_better": False
        }
    }
    file_path = tmp_path / "config_missing_field.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config_data, f)
    with pytest.raises(Exception):
        load_config(str(file_path))

def test_load_config_extra_fields(tmp_path):
    """Test that extra fields in the YAML config are ignored by the pydantic model."""
    config_data = {
        "model_id": "openai/whisper-small",
        "dataset_id": "mozilla-foundation/common_voice_en",
        "dataset_source": "HF",
        "language": "English",
        "repo_name": "whisper_en",
        "n_train_samples": 100,
        "n_test_samples": 50,
        "training_hp": {
            "push_to_hub": False,
            "hub_private_repo": True,
            "max_steps": 1000,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "learning_rate": 5e-5,
            "warmup_steps": 500,
            "gradient_checkpointing": True,
            "fp16": True,
            "eval_strategy": "steps",
            "per_device_eval_batch_size": 8,
            "predict_with_generate": False,
            "generation_max_length": 128,
            "save_steps": 100,
            "logging_steps": 50,
            "load_best_model_at_end": True,
            "save_total_limit": 3,
            "metric_for_best_model": "wer",
            "greater_is_better": False
        },
        "extra_field": "should be ignored"
    }
    file_path = tmp_path / "config_extra_field.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config_data, f)
    config_obj = load_config(str(file_path))
    # The extra field should not be present in the validated model output
    assert not hasattr(config_obj, "extra_field")

def test_load_config_invalid_yaml(tmp_path):
    """Test that load_config raises a yaml.YAMLError for an invalid YAML file."""
    file_path = tmp_path / "invalid.yaml"
    # Write an invalid YAML content
    file_path.write_text("model_id: openai/whisper-small: - invalid")
    with pytest.raises(yaml.YAMLError):
        load_config(str(file_path))
def test_load_config_non_dict(tmp_path):
    """Test that load_config raises an error when YAML file does not contain a dict."""
    # Writing YAML content that yields a list instead of dict
    non_dict_file = tmp_path / "non_dict.yaml"
    non_dict_file.write_text("- item1\n- item2")
    with pytest.raises(TypeError):
        load_config(str(non_dict_file))

def test_training_config_wrong_types():
    """Test that TrainingConfig raises a validation error when fields have wrong data types."""
    wrong_types_data = {
        "push_to_hub": "not a bool",  # wrong type
        "hub_private_repo": False,
        "max_steps": "1000",         # should be int
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": "5e-5",      # should be float
        "warmup_steps": 500,
        "gradient_checkpointing": True,
        "fp16": True,
        "eval_strategy": "steps",
        "per_device_eval_batch_size": 8,
        "predict_with_generate": False,
        "generation_max_length": 128,
        "save_steps": 100,
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "save_total_limit": 3,
        "metric_for_best_model": "wer",
        "greater_is_better": False
    }
    with pytest.raises(ValidationError):
        TrainingConfig(**wrong_types_data)

def test_load_config_extra_fields_in_training_hp(tmp_path):
    """Test that extra fields inside the nested training_hp are ignored by the pydantic model."""
    config_data = {
        "model_id": "openai/whisper-small",
        "dataset_id": "mozilla-foundation/common_voice_en",
        "dataset_source": "HF",
        "language": "English",
        "repo_name": "whisper_en",
        "n_train_samples": 100,
        "n_test_samples": 50,
        "training_hp": {
            "push_to_hub": False,
            "hub_private_repo": True,
            "max_steps": 1000,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "learning_rate": 5e-5,
            "warmup_steps": 500,
            "gradient_checkpointing": True,
            "fp16": True,
            "eval_strategy": "steps",
            "per_device_eval_batch_size": 8,
            "predict_with_generate": False,
            "generation_max_length": 128,
            "save_steps": 100,
            "logging_steps": 50,
            "load_best_model_at_end": True,
            "save_total_limit": 3,
            "metric_for_best_model": "wer",
            "greater_is_better": False,
            "extra_key": "should be ignored"
        }
    }
    file_path = tmp_path / "config_extra_traininghp.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config_data, f)
    config_obj = load_config(str(file_path))
    # Ensure that the extra key does not become an attribute on the TrainingConfig
    assert not hasattr(config_obj.training_hp, "extra_key")

def test_load_config_wrong_training_hp_structure(tmp_path):
    """Test that load_config raises a validation error when training_hp is not a dictionary."""
    config_data = {
        "model_id": "openai/whisper-small",
        "dataset_id": "mozilla-foundation/common_voice_en",
        "dataset_source": "HF",
        "language": "English",
        "repo_name": "whisper_en",
        "n_train_samples": 100,
        "n_test_samples": 50,
        "training_hp": "this should be a dict"
    }
    file_path = tmp_path / "config_wrong_traininghp.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config_data, f)
    with pytest.raises(Exception):
        load_config(str(file_path))