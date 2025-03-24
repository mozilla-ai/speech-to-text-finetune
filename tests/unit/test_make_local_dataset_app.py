from pathlib import Path
from typing import cast

import gradio
import pandas as pd
import pytest

from speech_to_text_finetune.make_custom_dataset_app import save_text_audio_to_file


@pytest.fixture
def dummy_audio_input():
    dummy_sample_rate = 1000
    dummy_audio_data = [0.0]
    return cast(gradio.Audio, (dummy_sample_rate, dummy_audio_data))


def test_save_text_audio_to_file_new_file(tmp_path, dummy_audio_input):
    test_sentence = "test"
    status, none = save_text_audio_to_file(
        audio_input=dummy_audio_input,
        sentence=test_sentence,
        dataset_dir=str(tmp_path),
        is_train_sample=True,
    )

    result_df = pd.read_csv(f"{tmp_path}/train/text.csv")
    assert Path(f"{tmp_path}/train/clips/rec_0.wav").is_file()
    assert result_df["sentence"][0] == test_sentence
    assert result_df["index"][0] == 0
    assert (
        status
        == f"✅ Updated {tmp_path}/train/text.csv \n✅ Saved recording to {tmp_path}/train/clips/rec_0.wav"
    )
    assert none is None


def test_save_text_audio_to_file_append_to_file(tmp_path, dummy_audio_input):
    test_sentence_2 = "test_2"
    save_text_audio_to_file(
        audio_input=dummy_audio_input,
        sentence="test_1",
        dataset_dir=tmp_path,
        is_train_sample=True,
    )
    status, none = save_text_audio_to_file(
        audio_input=dummy_audio_input,
        sentence=test_sentence_2,
        dataset_dir=tmp_path,
        is_train_sample=True,
    )

    result_df = pd.read_csv(f"{tmp_path}/train/text.csv")
    assert result_df["sentence"][1] == test_sentence_2
    assert Path(f"{tmp_path}/train/clips/rec_1.wav").is_file()
    assert (
        status
        == f"✅ Updated {tmp_path}/train/text.csv \n✅ Saved recording to {tmp_path}/train/clips/rec_1.wav"
    )
    assert none is None
