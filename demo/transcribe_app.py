import os
import gradio as gr
import spaces
from transformers import pipeline, Pipeline

is_hf_space = os.getenv("IS_HF_SPACE")
model_ids = [
    "",
    "mozilla-ai/whisper-small-gl (Galician)",
    "mozilla-ai/whisper-small-el (Greek)",
    "openai/whisper-tiny (Multilingual)",
    "openai/whisper-small (Multilingual)",
    "openai/whisper-medium (Multilingual)",
    "openai/whisper-large-v3 (Multilingual)",
    "openai/whisper-large-v3-turbo (Multilingual)",
]


def _load_local_model(model_dir: str) -> Pipeline:
    from transformers import (
        WhisperProcessor,
        WhisperTokenizer,
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
    )

    processor = WhisperProcessor.from_pretrained(model_dir)
    tokenizer = WhisperTokenizer.from_pretrained(model_dir, task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)

    return pipeline(
        task="automatic-speech-recognition",
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
    )


def _load_hf_model(model_repo_id: str) -> Pipeline:
    return pipeline(
        "automatic-speech-recognition",
        model=model_repo_id,
    )


@spaces.GPU
def transcribe(
    dropdown_model_id: str,
    hf_model_id: str,
    local_model_id: str,
    audio: gr.Audio,
) -> str:
    if dropdown_model_id and not hf_model_id and not local_model_id:
        dropdown_model_id = dropdown_model_id.split(" (")[0]
        pipe = _load_hf_model(dropdown_model_id)
    elif hf_model_id and not local_model_id and not dropdown_model_id:
        pipe = _load_hf_model(hf_model_id)
    elif local_model_id and not hf_model_id and not dropdown_model_id:
        pipe = _load_local_model(local_model_id)
    else:
        return "️️⚠️ Please select or fill at least and only one of the options above"
    text = pipe(audio)["text"]
    return text


def setup_gradio_demo():
    with gr.Blocks() as demo:
        gr.Markdown(
            """ # 🗣️ Speech-to-Text Transcription
            ### 1. Select which model to load from one of the options below.
            ### 2. Load the model by clicking the Load model button.
            ### 3. Record a message or upload an audio file.
            ### 4. Click Transcribe to see the transcription generated by the model.
            """
        )
        ### Model selection ###

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

        ### Transcription ###
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Record a message / Upload audio file",
            show_download_button=True,
            max_length=30,
        )
        transcribe_button = gr.Button("Transcribe")
        transcribe_output = gr.Text(label="Output")

        transcribe_button.click(
            fn=transcribe,
            inputs=[dropdown_model, user_model, local_model, audio_input],
            outputs=transcribe_output,
        )

    demo.launch()


if __name__ == "__main__":
    setup_gradio_demo()
