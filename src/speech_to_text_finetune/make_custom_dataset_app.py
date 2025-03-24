from pathlib import Path
from typing import Tuple
import gradio as gr
import soundfile as sf
import pandas as pd


def save_text_audio_to_file(
    audio_input: gr.Audio,
    sentence: str,
    dataset_dir: str,
    is_train_sample: bool,
) -> Tuple[str, None]:
    """
    Save the audio recording in a .wav file using the index of the text sentence in the filename.
    And save the associated text sentence in a .csv file using the same index.

    Args:
        audio_input (gr.Audio): Gradio audio object to be converted to audio data and then saved to a .wav file
        sentence (str): The text sentence that will be associated with the audio
        dataset_dir (str): The dataset directory path to store the indexed sentences and the associated audio files
        is_train_sample (bool): Whether to save the text-recording pair to the train or test dataset

    Returns:
        str: Status text for Gradio app
        None: Returning None here will reset the audio module to record again from scratch
    """
    Path(f"{dataset_dir}/train").mkdir(parents=True, exist_ok=True)
    Path(f"{dataset_dir}/train/clips").mkdir(parents=True, exist_ok=True)
    Path(f"{dataset_dir}/test").mkdir(parents=True, exist_ok=True)
    Path(f"{dataset_dir}/test/clips").mkdir(parents=True, exist_ok=True)

    data_path = (
        Path(f"{dataset_dir}/train/")
        if is_train_sample
        else Path(f"{dataset_dir}/test/")
    )
    text_path = Path(f"{data_path}/text.csv")
    if text_path.is_file():
        df = pd.read_csv(text_path)
    else:
        df = pd.DataFrame(columns=["index", "sentence"])

    index = len(df)
    text_df = pd.concat(
        [df, pd.DataFrame([{"index": index, "sentence": sentence}])],
        ignore_index=True,
    )
    text_df = text_df.sort_values(by="index")
    text_df.to_csv(text_path, index=False)

    audio_filepath = f"{data_path}/clips/rec_{index}.wav"

    sr, data = audio_input
    sf.write(file=audio_filepath, data=data, samplerate=sr)

    return (
        f"""✅ Updated {text_path} \n✅ Saved recording to {audio_filepath}""",
        None,
    )


def setup_gradio_demo():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # 🎤 Speech-to-text Dataset Recorder
            """
        )
        dataset_dir = gr.Text(
            value="local_data/custom", label="Select where to save the dataset"
        )

        local_sentence_textbox = gr.Text(label="Write your text here")

        local_audio_input = gr.Audio(sources=["microphone"], label="Record your voice")
        gr.Markdown("_Note: Make sure the recording is not longer than 30 seconds._")

        with gr.Row():
            with gr.Column():
                local_save_train_button = gr.Button(
                    "Save text-recording pair to Train dataset"
                )
                local_save_train_result = gr.Markdown()
            with gr.Column():
                local_save_test_button = gr.Button(
                    "Save text-recording pair to Test dataset"
                )
                local_save_test_result = gr.Markdown()

        # Need to pass str and bool values like gradio objects into the function args
        train_bool_gr = gr.Checkbox(True, visible=False)
        test_bool_gr = gr.Checkbox(False, visible=False)
        local_save_train_button.click(
            fn=save_text_audio_to_file,
            inputs=[
                local_audio_input,
                local_sentence_textbox,
                dataset_dir,
                train_bool_gr,
            ],
            outputs=[local_save_train_result, local_audio_input],
        )
        local_save_test_button.click(
            fn=save_text_audio_to_file,
            inputs=[
                local_audio_input,
                local_sentence_textbox,
                dataset_dir,
                test_bool_gr,
            ],
            outputs=[local_save_test_result, local_audio_input],
        )
    demo.launch()


if __name__ == "__main__":
    sentences = [""]
    setup_gradio_demo()
