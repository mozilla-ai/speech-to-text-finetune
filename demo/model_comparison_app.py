from typing import Tuple
import gradio as gr
from transformers import Pipeline
from demo.transcribe_app import model_ids, load_model


def transcribe(
    pipe: Pipeline, audio: gr.Audio, pipe_2: Pipeline | None
) -> Tuple[str, str]:
    text = pipe(audio)["text"]
    text_2 = pipe_2(audio)["text"] if pipe_2 else ""
    return text, text_2


def model_select_block():
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
        with gr.Column():
            local_model = gr.Textbox(
                label="Option 3: Paste local path to model directory",
                placeholder="artifacts/my-whisper-tiny",
            )

    load_model_button = gr.Button("Load model")
    model_loaded = gr.Markdown()
    return dropdown_model, user_model, local_model, load_model_button, model_loaded


def setup_gradio_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# 🗣️ Compare STT models")
        gr.Markdown("## Select baseline model")
        dropdown_model, user_model, local_model, load_model_button, model_loaded = (
            model_select_block()
        )
        gr.Markdown("## Select comparison model")
        (
            dropdown_model_2,
            user_model_2,
            local_model_2,
            load_model_button_2,
            model_loaded_2,
        ) = model_select_block()

        ### Transcription ###
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Record a message",
            show_download_button=True,
            max_length=30,
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
            inputs=[dropdown_model, user_model, local_model],
            outputs=[model, model_loaded],
        )
        load_model_button_2.click(
            fn=load_model,
            inputs=[dropdown_model_2, user_model_2, local_model_2],
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
