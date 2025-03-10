Get started with Speech-to-text-finetune Blueprint using one of the options below:
---

## Setup options

=== "☁️ Google Colab (GPU)"

      Finetune a STT model using CommonVoice data by launching the Google Colab notebook below

      Click the button below to launch the project directly in Google Colab:

      <p align="center"><a href="https://colab.research.google.com/github/mozilla-ai/speech-to-text-finetune/blob/main/demo/notebook.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" /></a></p>

=== "☁️ GitHub Codespaces"

      Click the button below to launch the project directly in GitHub Codespaces:

      <p align="center"><a href="https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=mozilla-ai/speech-to-text-finetune&skip_quickstart=true&machine=standardLinux32gb"><img src="https://github.com/codespaces/badge.svg" /></a></p>

      Once the Codespaces environment launches, inside the terminal, install dependencies:

      ```bash
      pip install -e .
      ```

      To load the app for making your own dataset for STT finetuning:

      ```bash
      python src/speech_to_text_finetune/make_custom_dataset_app.py
      ```

      To load the Transcription app:

      ```bash
      python demo/transcribe_app.py
      ```

      To run the Finetuning job:

      ```bash
      python src/speech_to_text_finetune/finetune_whisper.py
      ```

=== "💻 Local Installation"

      To install the project locally:

      ```bash
      git clone https://github.com/mozilla-ai/speech-to-text-finetune.git
      cd speech-to-text-finetune
      ```

      install dependencies:

      ```bash
      pip install -e .
      ```

      To load the app for making your own dataset for STT finetuning:

      ```bash
      python src/speech_to_text_finetune/make_custom_dataset_app.py
      ```

      To load the Transcription app:

      ```bash
      python demo/transcribe_app.py
      ```

      To run the Finetuning job:

      ```bash
      python src/speech_to_text_finetune/finetune_whisper.py
      ```



## Troubleshooting

Troubleshooting - TBA
