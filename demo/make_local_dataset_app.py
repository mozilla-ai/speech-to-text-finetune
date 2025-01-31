import shutil
from pathlib import Path
from typing import Tuple
import gradio as gr
import soundfile as sf
import pandas as pd
from huggingface_hub import hf_hub_download

from speech_to_text_finetune.hf_utils import get_available_languages_in_cv

parent_dir = "local_data"
dataset_id_cv = "mozilla-foundation/common_voice_17_0"
recorded_text_file = Path(f"{parent_dir}/text.csv")

languages_dict = get_available_languages_in_cv(dataset_id_cv)


def load_cv_sentences(language: str) -> str:
    language_id = languages_dict[language]
    source_text_file = Path(f"{parent_dir}/{language_id}_sentences.tsv")

    if not source_text_file.is_file():
        validated_sentences = hf_hub_download(
            repo_id=dataset_id_cv,
            filename=f"transcript/{language_id}/validated.tsv",
            repo_type="dataset",
            local_dir=parent_dir,
        )

        Path(validated_sentences).rename(source_text_file)
        shutil.rmtree(f"{parent_dir}/transcript")

    global sentences
    sentences = pd.read_table(source_text_file)["sentence"]
    return f"Loaded {language} sentences for transcription from {source_text_file}"


def load_from_index(index: int):
    return sentences[int(index)]


def go_previous(index: int) -> Tuple[int, str]:
    index -= 1
    return index, sentences[index]


def go_next(index: int) -> Tuple[int, str]:
    index += 1
    return index, sentences[index]


def save_audio_to_file(audio_input: gr.Audio, index: int) -> str:
    if recorded_text_file.is_file():
        text_df = pd.read_csv(recorded_text_file)
        text_df = pd.concat(
            [text_df, pd.DataFrame([{"index": index, "sentence": sentences[index]}])],
            ignore_index=True,
        )
        text_df = text_df.drop_duplicates().reset_index(drop=True)
    else:
        text_df = pd.DataFrame({"index": index, "sentence": [sentences[index]]})

    text_df = text_df.sort_values(by="index")
    text_df.to_csv(recorded_text_file, index=False)

    audio_filepath = f"{parent_dir}/rec_{index}.wav"

    sr, data = audio_input
    sf.write(file=audio_filepath, data=data, samplerate=sr)

    return f"Updated {recorded_text_file} and saved recording to {audio_filepath}"


def setup_gradio_demo():
    with gr.Blocks() as demo:
        ### Select language to build local dataset ###
        selected_lang = gr.Dropdown(
            choices=list(languages_dict.keys()), value="", label="Select a language"
        )
        load_lang_button = gr.Button("Load language text dataset")
        dataset_loaded = gr.Markdown()

        ### Dataset building ###
        index = gr.Number(value=0, label="Skip to index")
        sentence_textbox = gr.Text(label="Read and record the following sentence")

        load_index_button = gr.Button("Load from index")
        previous_sentence_button = gr.Button("Previous")
        next_sentence_button = gr.Button("Next")

        audio_input = gr.Audio(sources="microphone", label="Record")

        save_button = gr.Button("Save recording to file")
        save_result = gr.Markdown()

        ### Event listeners ###
        load_lang_button.click(
            fn=load_cv_sentences,
            inputs=[selected_lang, index],
            outputs=[dataset_loaded],
        )
        load_index_button.click(
            fn=load_from_index,
            inputs=[index],
            outputs=[sentence_textbox],
        )
        previous_sentence_button.click(
            fn=go_previous,
            inputs=[index],
            outputs=[index, sentence_textbox],
        )
        next_sentence_button.click(
            fn=go_next,
            inputs=[index],
            outputs=[index, sentence_textbox],
        )
        save_button.click(
            fn=save_audio_to_file,
            inputs=[audio_input, index],
            outputs=[save_result],
        )
    demo.launch()


if __name__ == "__main__":
    sentences = []
    setup_gradio_demo()
