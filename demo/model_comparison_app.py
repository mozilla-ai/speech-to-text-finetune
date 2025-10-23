from typing import Tuple

import gradio as gr
from transcribe_app import get_dropdown_model_ids, transcribe


def model_select_block() -> Tuple[gr.Dropdown, gr.Textbox, gr.Textbox]:
    with gr.Row():
        with gr.Column():
            dropdown_model = gr.Dropdown(
                choices=get_dropdown_model_ids(), label="Option 1: Select a model"
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

    return dropdown_model, user_model, local_model


def transcribe_sequentially(
    dropdown_model: str,
    user_model: str,
    local_model: str,
    dropdown_model_2: str,
    user_model_2: str,
    local_model_2: str,
    audio: gr.Audio,
) -> Tuple[str, str]:
    if text_1 := transcribe(dropdown_model, user_model, local_model, audio):
        yield text_1, ""
    if text_2 := transcribe(dropdown_model_2, user_model_2, local_model_2, audio):
        yield text_1, text_2


def setup_gradio_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# 🗣️ Compare STT models")
        gr.Markdown("## Select baseline model")
        dropdown_model, user_model, local_model = model_select_block()
        gr.Markdown("## Select comparison model")
        (
            dropdown_model_2,
            user_model_2,
            local_model_2,
        ) = model_select_block()

        gr.Markdown("## Record a sample or upload an audio file")
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

        transcribe_button.click(
            fn=transcribe_sequentially,
            inputs=[
                dropdown_model,
                user_model,
                local_model,
                dropdown_model_2,
                user_model_2,
                local_model_2,
                audio_input,
            ],
            outputs=[transcribe_output, transcribe_output_2],
        )

    demo.launch()


if __name__ == "__main__":
    setup_gradio_demo()
