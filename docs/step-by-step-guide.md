# **Step-by-Step Guide: How the Speech-to-Text-Finetune Blueprint Works**

This Blueprint enables you to fine-tune a Speech-to-Text (STT) model, using either your own data or the Common Voice dataset. This Step-by-Step guide you through the end-to-end process of finetuning an STT model based on your needs.

---

## **Overview**
This blueprint consists of three independent, yet complementary, components:

1. **Transcription app** 🎙️📝: A simple UI that lets you record your voice, pick any HF STT/ASR model, and get an instant transcription.

2. **Dataset maker app** 📂🎤: Another UI app that enables you to easily and quickly create your own Speech-to-Text dataset.

3. **Finetuning script** 🛠️🤖: A script to finetune your own STT model, either using Common Voice data or your own local data created by the Dataset maker app.

---

## Step-by-Step Guide

Visit the **[Getting Started](getting-started.md)** page for the initial project setup.

The following guide is a suggested user-flow for getting the most out of this Blueprint

### Step 1 - Initial transcription testing
Start by initially testing the quality of the Speech-to-Text models available in HuggingFace:

1. Simply execute:

    ```bash
    python demo/transcribe_app.py
    ```

2. Select or add the HF model id of your choice
3. Record a sample of your voice and get the transcribed text back. You may find that there are sometimes inaccuracies for your voice/accent/chosen language, indicating the model could benefit form finetuning on additional data.

### Step 2 - Make your Local Dataset for STT finetuning

1. Create your own, local dataset by running this command and following the instructions:

    ```bash
    python src/speech_to_text_finetune/make_local_dataset_app.py
    ```

2. Follow the instruction in the app to create at least 10 audio samples, which will be saved locally.

### Step 3 - Creating a finetuned STT model using your local data

1. Configure `config.yaml` with the model, local data directory and hyperparameters of your choice. Note that if you select `push_to_hub: True` you need to have an HF account and log in locally. For example:

    ```bash
    model_id: openai/whisper-tiny
    dataset_id: example_data/custom
    dataset_source: local
    language: English
    repo_name: default

    training_hp:
        push_to_hub: False
        hub_private_repo: True
        ...
    ```

2. Finetune a model by running:
```bash
python src/speech_to_text_finetune/finetune_whisper.py
```

> [!TIP]
> You can prematurely and gracefully stop the finetuning job by pressing CTRL+C. The rest of the code (evaluation, uploading the model) will execute as normal.

### Step 4 - (Optional) Creating a finetuned STT model using CommonVoice data
*Note: A Hugging Face account is required!*

1. Go to the Common Voice dataset repo and ask for explicit access request (should be approved instantly).
2. On Hugging Face create an Access Token
3. In your terminal, run the command `huggingface-cli login` and follow the instructions to log in to your account.
4. Configure `config.yaml` with the model, Common Voice dataset repo id of HF and hyperparameters of your choice. For example:
```bash
model_id = "openai/whisper-tiny"
dataset_id = "mozilla-foundation/common_voice_17_0"
language = "Greek"
repo_name: default

training_hp:
    push_to_hub: False
    hub_private_repo: True
    ...
```
5. Finetune a model by running:
```bash
python src/speech_to_text_finetune/finetune_whisper.py
```

### Step 5 - Evaluate transcription accuracy with your finetuned STT model
1. Start the Transcription app:
 ```bash
python demo/transcribe_app.py
```
2. Provided that `push_to_hub: True` when you Finetuned, you can select your HuggingFace model-id. If not, you can specify the local path to your model
3. Record a sample of your voice and get the transcribed text back.
4. You can easily switch between models with the same recorded sample to evaluate if the finetuned model has improved transcription accuracy.


### Step 6 - Compare transcription performance between two models

1. Start the Model Comparison app:
 ```bash
python demo/model_comparison_app.py
```
2. Select a baseline model, for example the model you used as a base for finetuning.
3. Select a comparison model, for example your finetuned model.
4. Record a sample of your voice and get two transcriptions back side-by-side for an easier manual evaluation.


## 🎨 **Customizing the Blueprint**

To better understand how you can tailor this Blueprint to suit your specific needs, please visit the **[Customization Guide](customization.md)**.

## 🤝 **Contributing to the Blueprint**

Want to help improve or extend this Blueprint? Check out the **[Future Features & Contributions Guide](future-features-contributions.md)** to see how you can contribute your ideas, code, or feedback to make this Blueprint even better!

## 📖 **Resources & References**

If you are interested in learning more about this topic, you might find the following resources helpful:
- [Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper) (Blog post by HuggingFace which inspired the implementation of the Blueprint!)

- [Automatic Speech Recognition Course from HuggingFace](https://huggingface.co/learn/audio-course/en/chapter5/introduction) (Series of Blog posts)

- [Fine-Tuning ASR Models: Key Definitions, Mechanics, and Use Cases](https://www.gladia.io/blog/fine-tuning-asr-models) (Blog post by Gladia)

- [Active Learning Approach for Fine-Tuning Pre-Trained ASR Model for a low-resourced Language](https://aclanthology.org/2023.icon-1.9.pdf) (Paper)

- [Exploration of Whisper fine-tuning strategies for low-resource ASR](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-024-00349-3) (Paper)

- [Finetuning Pretrained Model with Embedding of Domain and Language Information for ASR of Very Low-Resource Settings](https://www.worldscientific.com/doi/abs/10.1142/S2717554523500248?download=true&journalCode=ijalp) (Paper)
