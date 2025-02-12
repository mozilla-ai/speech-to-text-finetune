import json
from typing import Tuple, List
import gradio as gr
from transformers import pipeline, Pipeline
from speech_to_text_finetune.hf_utils import (
    validate_hf_model_id,
    get_available_languages_in_cv,
)

model_registry_file = "model_registry.json"


def load_model_registry() -> List:
    with open(model_registry_file) as json_file:
        model_registry = json.load(json_file)
    return list(model_registry)


model_ids = load_model_registry()
languages = get_available_languages_in_cv("mozilla-foundation/common_voice_17_0").keys()


def add_model(model_id: str) -> Tuple[gr.Dropdown, str]:
    if model_id in model_ids:
        status_text = f"Model {model_id} already in local registry"
    else:
        if validate_hf_model_id(model_id):
            with open(model_registry_file, "w") as json_file:
                model_ids.append(model_id)
                json.dump(model_ids, json_file, indent=4)
            status_text = f"Model {model_id} added to local registry"
        else:
            status_text = (
                f"Model {model_id} not found on Hugging Face. If you are certain the model exists, check "
                f"if you are logged in locally on Hugging Face or if you have the right permissions."
            )
    return gr.Dropdown(choices=model_ids, interactive=True), status_text


def remove_model(model_id: str) -> Tuple[gr.Dropdown, str]:
    with open(model_registry_file, "w") as json_file:
        model_ids.remove(model_id)
        json.dump(model_ids, json_file, indent=4)
    return gr.Dropdown(
        choices=model_ids, interactive=True
    ), f"Model {model_id} removed to local registry"


def load_model(model_id: str, language: str) -> Tuple[Pipeline, str]:
    if model_id and language:
        yield None, f"Loading {model_id}..."
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            generate_kwargs={"language": language},
        )
        yield pipe, f"Model {model_id} has been loaded."
    else:
        yield None, "Please select a model and a language from the dropdown"


def transcribe(pipe: Pipeline, audio: gr.Audio) -> str:
    text = pipe(audio)["text"]
    return text


def setup_gradio_demo():
    with gr.Blocks() as demo:
        ### Register your model ###
        model_to_register = gr.Textbox(
            label="Add a HuggingFace model ID to your local registry."
        )
        add_model_button = gr.Button("Add model")
        model_registered = gr.Markdown()

        ### Model & Language selection ###
        dropdown_model = gr.Dropdown(
            choices=model_ids, value=None, label="Select a model"
        )
        selected_lang = gr.Dropdown(
            choices=languages, value=None, label="Select a language"
        )
        remove_model_button = gr.Button("Remove model from registry")
        model_removed = gr.Markdown()
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

        add_model_button.click(
            fn=add_model,
            inputs=[model_to_register],
            outputs=[dropdown_model, model_registered],
        )

        load_model_button.click(
            fn=load_model,
            inputs=[dropdown_model, selected_lang],
            outputs=[model, model_loaded],
        )

        remove_model_button.click(
            fn=remove_model,
            inputs=[dropdown_model],
            outputs=[dropdown_model, model_removed],
        )

        transcribe_button.click(
            fn=transcribe, inputs=[model, audio_input], outputs=transcribe_output
        )

    demo.launch()


if __name__ == "__main__":
    setup_gradio_demo()
