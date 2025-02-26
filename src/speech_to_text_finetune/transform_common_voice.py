import argparse
from pathlib import Path

from pydub import AudioSegment
import pandas as pd
from loguru import logger


def transform_common_voice_to_local_dataset_format(
    cv_data_dir: str, output_dir: str
) -> None:
    cv_data_dir = Path(cv_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Transforming Common Voice at:\n\t {cv_data_dir}\nand exporting it to:\n\t{output_dir}"
    )

    clips_dir = cv_data_dir / "clips"
    other_df = pd.read_csv(cv_data_dir / "other.tsv", sep="\t")
    validated_df = pd.read_csv(cv_data_dir / "validated_sentences.tsv", sep="\t")

    # Map sentence_id to sentences to then use the sentence_id to pull the correct audio path from other.tsv
    sentence_map = {
        row["sentence_id"]: row["sentence"] for _, row in validated_df.iterrows()
    }

    local_text_data = []

    for i, row in other_df.iterrows():
        sentence_id = row["sentence_id"]
        audio_clip_path = Path(clips_dir) / row["path"]

        if sentence_id in sentence_map and Path(audio_clip_path).exists():
            local_text_data.append(f'{i},"{sentence_map[sentence_id]}"')

            # Convert .mp3 to .wav
            mp3_audio = AudioSegment.from_mp3(audio_clip_path)
            wav_path = output_dir / f"rec_{i}.wav"
            mp3_audio.export(wav_path, format="wav")

    text_csv_path = output_dir / "text.csv"
    with open(text_csv_path, "w") as f:
        f.write("index,sentence\n")
        f.write("\n".join(local_text_data))

    logger.info("Transformation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Define input / output directories")
    parser.add_argument(
        "cv_data_dir",
        type=str,
        help="Directory of Common Voice dataset, e.g. cv-corpus-20.0-delta-2024-12-06-el/cv-corpus-20.0-delta-2024-12-06/gl",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to store the transformed Common Voice dataset",
    )

    args = parser.parse_args()
    transform_common_voice_to_local_dataset_format(args.cv_data_dir, args.output_dir)
