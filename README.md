[![](https://dcbadge.limes.pink/api/server/YuMNeuKStr?style=flat)](https://discord.gg/YuMNeuKStr)
[![Docs](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/docs.yaml/)
[![Tests](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/tests.yaml/badge.svg)](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/tests.yaml/)
[![Ruff](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/lint.yaml/badge.svg?label=Ruff)](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/lint.yaml/)

<p align="center"><img src="./images/Blueprints-logo.png" width="35%" alt="Project logo"/></p>

This blueprint enables you to create your own [Speech-to-Text](https://en.wikipedia.org/wiki/Speech_recognition) / Automatic Speech Recognition (ASR) dataset, or use the [Common Voice](https://commonvoice.mozilla.org/) dataset, to finetune an ASR model to improve performance for your specific language & use-case. All of this can be done locally (even on your laptop!) ensuring no data leaves your machine, safeguarding your privacy.

📘 To explore this project further and discover other Blueprints, visit the [**Blueprints Hub**](https://developer-hub.mozilla.ai/blueprints/create-your-own-tailored-podcast-using-your-documents).

### 👉 📖 For more detailed guidance on using this project, please visit our [Docs here](https://mozilla-ai.github.io/Blueprint-template/)

### Built with
- Python 3.10+
- [Common Voice](https://commonvoice.mozilla.org)
- [Hugging Face](https://huggingface.co/)
- [Gradio](https://www.gradio.app/)

## Quick-start

1. Use a virtual environment and install dependencies: `pip install -e .` & install [ffmpeg](https://ffmpeg.org) e.g. for Ubuntu: `sudo apt install ffmpeg`, for Mac: `brew install ffmpeg`
2. (Optional) Create your own Speech-to-Text dataset by running `python demo/make_local_dataset_app.py`
3. Configure `config.yaml` with the model, dataset and hyperparameters of your choice.
4. Finetune Whisper by running `python src/speech_to_text_finetune/finetune_whisper.py`
5. Use the transcription app to test any HuggingFace STT model with your own voice/recordings by running `python demo/transcribe_app.py`


## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! To get started, you can check out the [CONTRIBUTING.md](CONTRIBUTING.md) file.
