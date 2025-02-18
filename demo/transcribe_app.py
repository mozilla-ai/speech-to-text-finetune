from pathlib import Path
from typing import Tuple
import gradio as gr
from transformers import pipeline, Pipeline
from speech_to_text_finetune.hf_utils import (
    get_available_languages_in_cv,
    hf_model_exists,
)


from speech_to_text_finetune.config import LANGUAGES_NAME_TO_ID

languages = LANGUAGES_NAME_TO_ID.keys()
model_ids = [
    "openai/whisper-tiny",
    "openai/whisper-small",
    "openai/whisper-medium",
    # Add here any other HF STT/ASR model id OR a local directory
    # "kostissz/whisper-tiny-el",  # custom HF model example
    # "artifacts/whisper-small-en",  # local model directory example
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
    if not hf_model_exists(model_repo_id):
        return (
            None,
            f"⚠️ Couldn't find {model_repo_id} on Hugging Face. If its a private repo, make sure you are logged in locally.",
        )
    return pipeline(
        "automatic-speech-recognition",
        model=model_repo_id,
        generate_kwargs={"language": language},
    ), f"✅ HF Model {model_repo_id} has been loaded."


def load_model(model_id: str, language: str, local: bool) -> Tuple[Pipeline, str]:
    if model_id and language:
        yield None, f"Loading {model_id}..."
        yield (
            _load_local_model(model_id, language)
            if local
            else _load_hf_model(model_id, language)
        )
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
        with gr.Row():
            with gr.Column():
                dropdown_model = gr.Dropdown(
                    choices=model_ids, value=None, label="Select a model"
                )
            with gr.Column():
                local_check = gr.Checkbox(label="Local model")

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
            inputs=[dropdown_model, selected_lang, local_check],
            outputs=[model, model_loaded],
        )

        transcribe_button.click(
            fn=transcribe, inputs=[model, audio_input], outputs=transcribe_output
        )

    demo.launch()


if __name__ == "__main__":
    setup_gradio_demo()
