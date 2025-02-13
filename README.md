[![](https://dcbadge.limes.pink/api/server/YuMNeuKStr?style=flat)](https://discord.gg/YuMNeuKStr)
[![Docs](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/docs.yaml/)
[![Tests](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/tests.yaml/badge.svg)](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/tests.yaml/)
[![Ruff](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/lint.yaml/badge.svg?label=Ruff)](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/lint.yaml/)

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

This blueprint enables you to create your own [Speech-to-Text](https://en.wikipedia.org/wiki/Speech_recognition) / Automatic Speech Recognition (ASR) dataset and model to improve performance of standard STT models for your specific language & use-case. All of this can be done locally (even on your laptop!) ensuring no data leaves your machine, safeguarding your privacy. You can choose to finetune a model either on your own, local speech-to-text data or use the Common Voice dataset. Using Common Voice enables this blueprint to support an impressively wide variety of languages! More the exact list of languages supported please visit the Common Voice [website](https://commonvoice.mozilla.org/en/languages).

📖 For more detailed guidance on using this project, please visit our [Docs here](https://mozilla-ai.github.io/speech-to-text-finetune/)

📘 To explore this project further and discover other Blueprints, visit the [**Blueprints Hub**](https://developer-hub.mozilla.ai/).

### Built with
- Python 3.10+
- [Hugging Face](https://huggingface.co/)
- [Gradio](https://www.gradio.app/)
- [Common Voice](https://commonvoice.mozilla.org)

## Quick-start

**_Note_**: All scripts should be executed from the root directory of the repository.

This blueprint consists of three independent, yet complementary, components:

1. **Transcription app**: A simple UI that lets you record your voice, pick any HF STT/ASR model, and get an instant transcription.
2. **Dataset maker app**: Another UI app that enables you to easily and quickly create your own Speech-to-Text dataset.
3. **Finetuning script**: A script to finetune your own STT model, either using Common Voice data or your own local data created by the Dataset maker app.

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

### Making your own STT model using Common Voice

**_Note_**: A Hugging Face account is required!

1. Go to the Common Voice dataset [repo](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) and ask for explicit access request (should be approved instantly).
2. On Hugging Face create an [Access Token](https://huggingface.co/docs/hub/en/security-tokens)
3. In your terminal, run the command `huggingface-cli login` and follow the instructions to log in to your account.
4. Configure `config.yaml` with the model, Common Voice dataset repo id of HF and hyperparameters of your choice.
5. Finetune a model by running: `python src/speech_to_text_finetune/finetune_whisper.py`


## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! To get started, you can check out the [CONTRIBUTING.md](CONTRIBUTING.md) file.
