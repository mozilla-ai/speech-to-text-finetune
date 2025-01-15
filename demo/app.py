from typing import Tuple

import gradio as gr
from transformers import pipeline, Pipeline

languages = ["greek", "galician"]
model_ids = [
    "kostissz/whisper-tiny-gl",
    "kostissz/whisper-tiny-el",
    "openai/whisper-small",
    "openai/whisper-tiny",
]


def load_model(model_id: str, language: str) -> Tuple[Pipeline, str]:
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        generate_kwargs={"language": language},
    )
    return pipe, f"Model {model_id} has been loaded."


def transcribe(pipe: Pipeline, audio: gr.Audio) -> str:
    text = pipe(audio)["text"]
    return text


def setup_gradio_demo():
    with gr.Blocks() as demo:
        ### Model & Language selection ###
        dropdown_model = gr.Dropdown(
            choices=model_ids, value=None, label="Select a model"
        )
        gr.Markdown("Or")
        user_input_model = gr.Textbox(
            label="Input a HF model id of a finetuned whisper model, e.g. openai/whisper-large-v3"
        )
        gr.Markdown("Next")
        selected_lang = gr.Dropdown(
            choices=languages, value=None, label="Select a language"
        )
        load_model_button = gr.Button("Load model")
        model_loaded = gr.Markdown()

        ### Transcription ###
        audio_input = gr.Audio(
            sources="microphone", type="filepath", label="Record a message"
        )
        transcribe_button = gr.Button("Transcribe")
        transcribe_output = gr.Text(label="Output")

        ### Event listeners ###
        model = gr.State()
        selected_model = user_input_model if user_input_model else dropdown_model
        load_model_button.click(
            fn=load_model,
            inputs=[selected_model, selected_lang],
            outputs=[model, model_loaded],
        )

        transcribe_button.click(
            fn=transcribe, inputs=[model, audio_input], outputs=transcribe_output
        )

    demo.launch()


if __name__ == "__main__":
    setup_gradio_demo()
