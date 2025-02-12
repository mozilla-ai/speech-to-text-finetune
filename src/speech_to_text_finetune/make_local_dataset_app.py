import shutil
from pathlib import Path
from typing import Tuple
import gradio as gr
import soundfile as sf
import pandas as pd
from huggingface_hub import hf_hub_download

from speech_to_text_finetune.hf_utils import get_available_languages_in_cv

parent_dir = "local_data"
cv_dir = f"{parent_dir}/cv"
custom_dir = f"{parent_dir}/custom"
dataset_id_cv = "mozilla-foundation/common_voice_17_0"

languages_dict = get_available_languages_in_cv(dataset_id_cv)


def load_cv_sentences(language: str) -> str:
    """
    Loads all the validated text sentences from the Common Voice dataset of the language requested.
    If the file doesn't exist locally, it will download it from HF.

    Args:
        language (str): Full name of the language to download the text data from.
        Needs to be supported by Common Voice.

    Returns:
        str: Status text for Gradio app
    """
    language_id = languages_dict[language]
    Path(cv_dir).mkdir(parents=True, exist_ok=True)
    source_text_file = Path(f"{cv_dir}/{language_id}_sentences.tsv")

    if not source_text_file.is_file():
        validated_sentences = hf_hub_download(
            repo_id=dataset_id_cv,
            filename=f"transcript/{language_id}/validated.tsv",
            repo_type="dataset",
            local_dir=cv_dir,
        )

        Path(validated_sentences).rename(source_text_file)
        shutil.rmtree(f"{cv_dir}/transcript")

    global sentences
    sentences = pd.read_table(source_text_file)["sentence"]
    return f"✅ Loaded {language} sentences from {source_text_file}"


def load_from_index(index: int):
    return sentences[int(index)]


def go_previous(index: int) -> Tuple[int, str]:
    index -= 1
    return index, sentences[index]


def go_next(index: int) -> Tuple[int, str]:
    index += 1
    return index, sentences[index]


def save_text_audio_to_file(
    audio_input: gr.Audio,
    sentence: str,
    dataset_dir: str,
    index: str | None = None,
) -> Tuple[str, None]:
    """
    Save the audio recording in a .wav file using the index of the text sentence in the filename.
    And save the associated text sentence in a .csv file using the same index.

    Args:
        audio_input (gr.Audio): Gradio audio object to be converted to audio data and then saved to a .wav file
        sentence (str): The text sentence that will be associated with the audio
        dataset_dir (str): The dataset directory path to store the indexed sentences and the associated audio files
        index (str | None): Index of the text sentence that will be associated with the audio.
        If None, start from 0 or append after the last element in the existing .csv

    Returns:
        str: Status text for Gradio app
    """
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    if Path(f"{dataset_dir}/text.csv").is_file():
        text_df = pd.read_csv(f"{dataset_dir}/text.csv")
        if index is None:
            index = len(text_df)
        text_df = pd.concat(
            [text_df, pd.DataFrame([{"index": index, "sentence": sentence}])],
            ignore_index=True,
        )
        text_df = text_df.drop_duplicates().reset_index(drop=True)
    else:
        if index is None:
            index = 0
        text_df = pd.DataFrame({"index": index, "sentence": [sentence]})

    text_df = text_df.sort_values(by="index")
    text_df.to_csv(f"{dataset_dir}/text.csv", index=False)

    audio_filepath = f"{dataset_dir}/rec_{index}.wav"

    sr, data = audio_input
    sf.write(file=audio_filepath, data=data, samplerate=sr)

    return (
        f"""✅ Updated {dataset_dir}/text.csv \n✅ Saved recording to {audio_filepath}""",
        None,
    )


def setup_gradio_demo():
    custom_css = ".gradio-container { max-width: 450px; margin: 0 auto; }"
    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown(
            """
            # 🎤 Speech-to-text Dataset Recorder
            ### 1. Select your language
            """
        )
        selected_lang = gr.Dropdown(
            choices=list(languages_dict.keys()), value="", label="Select a language"
        )
        gr.Markdown(
            "### 2. Click the Common Voice tab to use text data from the Common Voice dataset "
            "**_OR_** Click the Custom data to write your own text data"
        )
        with gr.Row():
            with gr.Tab("Common Voice"):
                gr.Markdown("### 3. Click **Load language text dataset**.")
                load_lang_button = gr.Button("Load Common Voice text dataset")
                dataset_loaded = gr.Markdown()

                gr.Markdown(
                    "### 4. Set an index and click **Load from index** or use **← Previous** / **Next →** to navigate sentences."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        index = gr.Number(value=0, label="Skip to index")
                    with gr.Column(scale=3):
                        load_index_button = gr.Button("Load from index")
                        with gr.Row():
                            previous_sentence_button = gr.Button("← Previous")
                            next_sentence_button = gr.Button("Next →")

                cv_sentence_textbox = gr.Text(
                    label="Read and record the following sentence"
                )
                gr.Markdown(
                    "_Note: Make sure the recording is not longer than 30 seconds._"
                )
                cv_audio_input = gr.Audio(
                    sources=["microphone"], label="Record your voice"
                )
                cv_save_button = gr.Button("Save text-recording pair to file")
                cv_save_result = gr.Markdown()

            with gr.Tab("Custom data"):
                gr.Markdown("""
                ### 3. Write your text in the box below.
                ### 4. Record audio and click **Save text-recording pair to file**.
                """)

                own_sentence_textbox = gr.Text(label="Write your text here")

                own_audio_input = gr.Audio(
                    sources=["microphone"], label="Record your voice"
                )
                gr.Markdown(
                    "_Note: Make sure the recording is not longer than 30 seconds._"
                )

                own_save_button = gr.Button("Save text-recording pair to file")
                own_save_result = gr.Markdown()

        load_lang_button.click(
            fn=load_cv_sentences,
            inputs=[selected_lang],
            outputs=[dataset_loaded],
        )
        load_index_button.click(
            fn=load_from_index,
            inputs=[index],
            outputs=[cv_sentence_textbox],
        )
        previous_sentence_button.click(
            fn=go_previous,
            inputs=[index],
            outputs=[index, cv_sentence_textbox],
        )
        next_sentence_button.click(
            fn=go_next,
            inputs=[index],
            outputs=[index, cv_sentence_textbox],
        )
        cv_dir_gr = gr.Text(cv_dir, visible=False)
        cv_save_button.click(
            fn=save_text_audio_to_file,
            inputs=[cv_audio_input, cv_sentence_textbox, cv_dir_gr, index],
            outputs=[cv_save_result, cv_audio_input],
        )
        custom_dir_gr = gr.Text(custom_dir, visible=False)
        own_save_button.click(
            fn=save_text_audio_to_file,
            inputs=[own_audio_input, own_sentence_textbox, custom_dir_gr],
            outputs=[own_save_result, own_audio_input],
        )
    demo.launch()


if __name__ == "__main__":
    sentences = [""]
    setup_gradio_demo()
