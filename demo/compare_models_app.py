import os
from pathlib import Path
from typing import Tuple
import gradio as gr
from transformers import pipeline, Pipeline
from huggingface_hub import repo_exists


from speech_to_text_finetune.config import LANGUAGES_NAME_TO_ID

is_hf_space = os.getenv("IS_HF_SPACE")
languages = LANGUAGES_NAME_TO_ID.keys()
model_ids = [
    "",
    "openai/whisper-tiny",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
]


def _load_local_model(model_dir: str, language: str) -> Tuple[Pipeline | None, str]:
    if not Path(model_dir).is_dir():
        return None, f"⚠️ Couldn't find local model directory: {model_dir}"
    from transformers import (
        WhisperProcessor,
        WhisperTokenizer,
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
    )

    processor = WhisperProcessor.from_pretrained(model_dir)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_dir, language=language, task="transcribe"
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)

    return pipeline(
        task="automatic-speech-recognition",
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
    ), f"✅ Local model has been loaded from {model_dir}."


def _load_hf_model(model_repo_id: str, language: str) -> Tuple[Pipeline | None, str]:
    if not repo_exists(model_repo_id):
        return (
            None,
            f"⚠️ Couldn't find {model_repo_id} on Hugging Face. If its a private repo, make sure you are logged in locally.",
        )
    return pipeline(
        "automatic-speech-recognition",
        model=model_repo_id,
        generate_kwargs={"language": language},
    ), f"✅ HF Model {model_repo_id} has been loaded."


def load_model(
    language: str, dropdown_model_id: str, hf_model_id: str, local_model_id: str
) -> Tuple[Pipeline, str]:
    if dropdown_model_id and not hf_model_id and not local_model_id:
        yield None, f"Loading {dropdown_model_id}..."
        yield _load_hf_model(dropdown_model_id, language)
    elif hf_model_id and not local_model_id and not dropdown_model_id:
        yield None, f"Loading {hf_model_id}..."
        yield _load_hf_model(hf_model_id, language)
    elif local_model_id and not hf_model_id and not dropdown_model_id:
        yield None, f"Loading {local_model_id}..."
        yield _load_local_model(local_model_id, language)
    else:
        yield (
            None,
            "️️⚠️ Please select or fill at least and only one of the options above",
        )
    if not language:
        yield None, "⚠️ Please select a language from the dropdown"


def transcribe(
    pipe: Pipeline, audio: gr.Audio, pipe_2: Pipeline | None
) -> Tuple[str, str]:
    text = pipe(audio)["text"]
    text_2 = pipe_2(audio)["text"] if pipe_2 else ""
    return text, text_2


def setup_gradio_demo():
    with gr.Blocks() as demo:
        gr.Markdown(
            """ # 🗣️ Speech-to-Text Transcription
            ### 1. Select a language from the dropdown menu.
            ### 2. Select which model to load from one of the options below.
            ### 3. Load the model by clicking the Load model button.
            ### 4. Record a message and click Transcribe to see the transcription.
            """
        )
        ### Language & Model selection ###

        selected_lang = gr.Dropdown(
            choices=list(languages), value=None, label="Select a language"
        )

        with gr.Row():
            with gr.Column():
                dropdown_model = gr.Dropdown(
                    choices=model_ids, label="Option 1: Select a model"
                )
            with gr.Column():
                user_model = gr.Textbox(
                    label="Option 2: Paste HF model id",
                    placeholder="my-username/my-whisper-tiny",
                )
            with gr.Column(visible=not is_hf_space):
                local_model = gr.Textbox(
                    label="Option 3: Paste local path to model directory",
                    placeholder="artifacts/my-whisper-tiny",
                )

        load_model_button = gr.Button("Load model")
        model_loaded = gr.Markdown()
        with gr.Row():
            with gr.Column():
                dropdown_model_2 = gr.Dropdown(
                    choices=model_ids, label="Option 1: Select a model"
                )
            with gr.Column():
                user_model_2 = gr.Textbox(
                    label="Option 2: Paste HF model id",
                    placeholder="my-username/my-whisper-tiny",
                )
            with gr.Column(visible=not is_hf_space):
                local_model_2 = gr.Textbox(
                    label="Option 3: Paste local path to model directory",
                    placeholder="artifacts/my-whisper-tiny",
                )

        load_model_button_2 = gr.Button("Load model")
        model_loaded_2 = gr.Markdown()

        ### Transcription ###
        audio_input = gr.Audio(
            sources=["microphone"], type="filepath", label="Record a message"
        )
        transcribe_button = gr.Button("Transcribe")
        with gr.Row():
            with gr.Column():
                transcribe_output = gr.Text(label="Output of primary model")
            with gr.Column():
                transcribe_output_2 = gr.Text(label="Output of comparison model")

        ### Event listeners ###
        model = gr.State()
        model_2 = gr.State()
        load_model_button.click(
            fn=load_model,
            inputs=[selected_lang, dropdown_model, user_model, local_model],
            outputs=[model, model_loaded],
        )
        load_model_button_2.click(
            fn=load_model,
            inputs=[selected_lang, dropdown_model_2, user_model_2, local_model_2],
            outputs=[model_2, model_loaded_2],
        )
        transcribe_button.click(
            fn=transcribe,
            inputs=[model, audio_input, model_2],
            outputs=[transcribe_output, transcribe_output_2],
        )

    demo.launch()


if __name__ == "__main__":
    setup_gradio_demo()
