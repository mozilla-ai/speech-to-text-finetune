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

This blueprint enables you to create your own [Speech-to-Text](https://en.wikipedia.org/wiki/Speech_recognition) / Automatic Speech Recognition (ASR) dataset, or use the [Common Voice](https://commonvoice.mozilla.org/) dataset, to finetune an ASR model to improve performance for your specific language & use-case. All of this can be done locally (even on your laptop!) ensuring no data leaves your machine, safeguarding your privacy. Using Common Voice as a backbone enables this blueprint to support an impressively wide variety of languages! More the exact list of languages supported please visit the Common Voice [website](https://commonvoice.mozilla.org/en/languages).

📘 To explore this project further and discover other Blueprints, visit the [**Blueprints Hub**](https://developer-hub.mozilla.ai/).

 📖 For more detailed guidance on using this project, please visit our [Docs here](https://mozilla-ai.github.io/speech-to-text-finetune/)

### Built with
- Python 3.10+
- [Common Voice](https://commonvoice.mozilla.org)
- [Hugging Face](https://huggingface.co/)
- [Gradio](https://www.gradio.app/)

## Quick-start

**_Note_**: All scripts should be executed from the root directory of the repository.

This blueprint consists of three independent, yet complementary, components:

1. **Transcription app**: A simple UI that lets you record your voice, pick any HF ASR model, and get an instant transcription.
2. **Dataset maker app**: Another UI app that enables you to easily and quickly create your own Speech-to-Text dataset.
3. **Finetuning script**: A script to finetune your own STT model, either using Common Voice data or your own local data created by the Dataset maker app.

### Suggested flow for this repository

1. Use a virtual environment and install dependencies: `pip install -e .` & [ffmpeg](https://ffmpeg.org) e.g. for Ubuntu: `sudo apt install ffmpeg`, for Mac: `brew install ffmpeg`
2. Try existing transcription HF models on your own language & voice locally: `python demo/transcribe_app.py`
3. If you are not happy with the results, you can finetune a model with data of your language from Common Voice
   1. Configure `config.yaml` with the model, Common Voice dataset id from HF and hyperparameters of your choice.
   2. Finetune a model: `python src/speech_to_text_finetune/finetune_whisper.py`
4. Try again the transcription app with your newly finetuned model.
5. If the results are still not satisfactory, create your own Speech-to-Text dataset and model.
   1. Create a dataset: `python demo/make_local_dataset_app.py`
   2. Configure `config.yaml` with the model, local data directory and hyperparameters of your choice.
   3. Finetune a model: `python src/speech_to_text_finetune/finetune_whisper.py`
6. Finally try again the transcription app with the new model finetuned specifically for your own voice!

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! To get started, you can check out the [CONTRIBUTING.md](CONTRIBUTING.md) file.
