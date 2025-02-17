import os
import shutil
from pathlib import Path

from speech_to_text_finetune.config import load_config
from speech_to_text_finetune.finetune_whisper import run_finetuning


def test_finetune_whisper_local():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    cfg_path = f"{dir_path}/config.yaml"

    base_results, eval_results = run_finetuning(config_path=cfg_path)

    cfg = load_config(cfg_path)
    expected_dir_path = Path(f"artifacts/{cfg.repo_name}")
    assert expected_dir_path.exists()

    assert 0 < base_results["eval_loss"] < 10
    assert 0 < base_results["eval_wer"] < 100
    assert 0 < eval_results["eval_loss"] < 10
    assert 0 < eval_results["eval_wer"] < 100

    shutil.rmtree(expected_dir_path)
