from pathlib import Path
from typing import Tuple
import gradio as gr
import soundfile as sf
import pandas as pd

parent_dir = "local_data"
custom_dir = f"{parent_dir}/custom"


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
            """
        )
        local_sentence_textbox = gr.Text(label="Write your text here")

        local_audio_input = gr.Audio(sources=["microphone"], label="Record your voice")
        gr.Markdown("_Note: Make sure the recording is not longer than 30 seconds._")

        local_save_button = gr.Button("Save text-recording pair to file")
        local_save_result = gr.Markdown()

        custom_dir_gr = gr.Text(custom_dir, visible=False)
        local_save_button.click(
            fn=save_text_audio_to_file,
            inputs=[local_audio_input, local_sentence_textbox, custom_dir_gr],
            outputs=[local_save_result, local_audio_input],
        )
    demo.launch()


if __name__ == "__main__":
    sentences = [""]
    setup_gradio_demo()
