from typing import Tuple
import gradio as gr
from transformers import pipeline, Pipeline
from speech_to_text_finetune.hf_utils import get_available_languages_in_cv


languages = get_available_languages_in_cv("mozilla-foundation/common_voice_17_0").keys()
model_ids = [
    "kostissz/whisper-tiny-gl",
    "kostissz/whisper-tiny-el",
    "openai/whisper-tiny",
    "openai/whisper-small",
    "openai/whisper-medium",
]


def load_model(model_id: str, language: str) -> Tuple[Pipeline, str]:
    if model_id and language:
        yield None, f"Loading {model_id}..."
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            generate_kwargs={"language": language},
        )
        yield pipe, f"✅ Model {model_id} has been loaded."
    else:
        yield None, "⚠️ Please select a model and a language from the dropdown"


def transcribe(pipe: Pipeline, audio: gr.Audio) -> str:
    text = pipe(audio)["text"]
    return text


def setup_gradio_demo():
    with gr.Blocks() as demo:
        gr.Markdown(
            """ # 🗣️ Speech-to-Text Transcription
            ### 1. Select a model and a language from the dropdowns.
            ### 2. Load the model by clicking the Load model button.
            ### 3. Record a message and click Transcribe to see the transcription.
            """
        )
        ### Model & Language selection ###
        dropdown_model = gr.Dropdown(
            choices=model_ids, value=None, label="Select a model"
        )
        selected_lang = gr.Dropdown(
            choices=list(languages), value=None, label="Select a language"
        )
        load_model_button = gr.Button("Load model")
        model_loaded = gr.Markdown()

        ### Transcription ###
        audio_input = gr.Audio(
            sources=["microphone"], type="filepath", label="Record a message"
        )
        transcribe_button = gr.Button("Transcribe")
        transcribe_output = gr.Text(label="Output")

        ### Event listeners ###
        model = gr.State()
        load_model_button.click(
            fn=load_model,
            inputs=[dropdown_model, selected_lang],
            outputs=[model, model_loaded],
        )

        transcribe_button.click(
            fn=transcribe, inputs=[model, audio_input], outputs=transcribe_output
        )

    demo.launch()


if __name__ == "__main__":
    setup_gradio_demo()
