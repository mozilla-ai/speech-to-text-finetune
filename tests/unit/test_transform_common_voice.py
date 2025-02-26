import csv
from pathlib import Path

from speech_to_text_finetune.transform_common_voice import (
    transform_common_voice_to_local_dataset_format,
)


def test_transform_common_voice_to_local_dataset_format(
    example_common_voice_data, tmp_path
):
    transform_common_voice_to_local_dataset_format(
        cv_data_dir=example_common_voice_data, output_dir=str(tmp_path)
    )

    # Verify generated text.csv has the expected format
    expected_text_csv_content = "index,sentence\n0,Example sentence"

    with open(tmp_path / "text.csv", "r") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        lines = list(csv_reader)[:]  # Skip the header row
        output_text_csv_content = "\n".join([",".join(line) for line in lines])

    assert expected_text_csv_content == output_text_csv_content
    assert Path(tmp_path / "rec_0.wav").is_file()
