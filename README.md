<p align="center">
  <picture>
    <!-- When the user prefers dark mode, show the white logo -->
    <source media="(prefers-color-scheme: dark)" srcset="./images/Blueprint-logo-white.png">
    <!-- When the user prefers light mode, show the black logo -->
    <source media="(prefers-color-scheme: light)" srcset="./images/Blueprint-logo-black.png">
    <!-- Fallback: default to the black logo -->
    <img src="./images/Blueprint-logo-black.png" width="35%" alt="Project logo"/>
  </picture>
</p>

# Finetuning Speech-to-Text models: a Blueprint by Mozilla.ai for building your own STT/ASR dataset & model

[![](https://dcbadge.limes.pink/api/server/YuMNeuKStr?style=flat)](https://discord.gg/YuMNeuKStr)
[![Docs](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/docs.yaml/)
[![Tests](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/tests.yaml/badge.svg)](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/tests.yaml/)
[![Ruff](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/lint.yaml/badge.svg?label=Ruff)](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/lint.yaml/)


This blueprint enables you to create your own [Speech-to-Text](https://en.wikipedia.org/wiki/Speech_recognition) / Automatic Speech Recognition (ASR) dataset and model to improve performance of standard STT models for your specific language & use-case. All of this can be done locally (even on your laptop!) ensuring no data leaves your machine, safeguarding your privacy. You can choose to finetune a model either on your own, local speech-to-text data or use the Common Voice dataset. Using Common Voice enables this blueprint to support an impressively wide variety of languages! More the exact list of languages supported please visit the Common Voice [website](https://commonvoice.mozilla.org/en/languages).

<img src="./images/speech-to-text-finetune-diagram.png" width="1200" alt="speech-to-text-finetune Diagram" />

## Example Results

Input Speech audio: 

Text output:

| [openai/whisper-small](https://huggingface.co/openai/whisper-small) | [mozilla-ai/whisper-small-el](https://huggingface.co/kostissz/whisper-small-el) * | mozilla-ai/whisper-small-el-plus-local ** |
| -------------| ------------------- | ----------------- |
| -------------| ------------------- | ----------------- |

\* Finetuned on the Greek set Common Voice 17.0

\** Finetuned on the Greek set Common Voice 17.0 + 16 locally-recorded, custom samples

### 📖 For more detailed guidance on using this project, please visit our [Docs here](https://mozilla-ai.github.io/speech-to-text-finetune/)

📘 To explore this project further and discover other Blueprints, visit the [**Blueprints Hub**](https://developer-hub.mozilla.ai/).

### Built with

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%F0%9F%A4%97-yellow)](https://huggingface.co/) [![Gradio](https://img.shields.io/badge/Gradio-%F0%9F%8E%A8-green)](https://www.gradio.app/) [![Common Voice](https://img.shields.io/badge/Common%20Voice-%F0%9F%8E%A4-orange)](https://commonvoice.mozilla.org)

📖 For more detailed guidance on using this project, please visit our [Docs here](https://mozilla-ai.github.io/speech-to-text-finetune/)

## Quick-start

Try out already finetuned models with our transcription app:

| Google Colab | HuggingFace Spaces  | GitHub Codespaces |
| -------------| ------------------- | ----------------- |
| [![Try Finetuning on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mozilla-ai/speech-to-text-finetune/blob/main/demo/notebook.ipynb) | _Coming Soon!_ | [![Try on Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=mozilla-ai/speech-to-text-finetune&skip_quickstart=true&machine=standardLinux32gb) |


## Try it locally

### Setup

1. Use a virtual environment and install dependencies: `pip install -e .` & [ffmpeg](https://ffmpeg.org) e.g. for Ubuntu: `sudo apt install ffmpeg`, for Mac: `brew install ffmpeg`

### Evaluate existing STT models from the HuggingFace repository.

1. Simply execute: `python demo/transcribe_app.py`
2. Add the HF model id of your choice
3. Record a sample of your voice and get the transcribe text back

### Making your own STT model using Local Data

1. Create your own, local dataset by running this command and following the instructions: `python src/speech_to_text_finetune/make_local_dataset_app.py`
2. Configure `config.yaml` with the model, local data directory and hyperparameters of your choice. Note that if you select `push_to_hub: True` you need to have an HF account and log in locally.
3. Finetune a model by running: `python src/speech_to_text_finetune/finetune_whisper.py`
4. Test the finetuned model in the transcription app: `python demo/transcribe_app.py`

### Making your own STT model using Common Voice

**_Note_**: A Hugging Face account is required!

1. Go to the Common Voice dataset [repo](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) and ask for explicit access request (should be approved instantly).
2. On Hugging Face create an [Access Token](https://huggingface.co/docs/hub/en/security-tokens)
3. In your terminal, run the command `huggingface-cli login` and follow the instructions to log in to your account.
4. Configure `config.yaml` with the model, Common Voice dataset repo id of HF and hyperparameters of your choice.
5. Finetune a model by running: `python src/speech_to_text_finetune/finetune_whisper.py`
6. Test the finetuned model in the transcription app: `python demo/transcribe_app.py`

## Troubleshooting

If you are having issues / bugs, check our [Troubleshooting](https://mozilla-ai.github.io/speech-to-text-finetune/getting-started/#troubleshooting) section, before opening a new issue.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! To get started, you can check out the [CONTRIBUTING.md](CONTRIBUTING.md) file.
