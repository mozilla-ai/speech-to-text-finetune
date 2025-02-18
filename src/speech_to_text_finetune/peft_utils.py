import os

import evaluate
import torch
from datasets import Audio
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration
import gc
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


class SavePeftModelCallback(TrainerCallback):
    # This callback helps to save only the adapter weights and remove the base model weights.
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


def load_whisper_peft(model_id: str):
    peft_config = PeftConfig.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, model_id)
    model.config.use_cache = True
    return model


def evaluate_w_peft(model, test_dataset, data_collator, processor, language):
    metric = evaluate.load("wer")
    eval_dataloader = DataLoader(
        test_dataset, batch_size=8, collate_fn=data_collator, num_workers=0
    )
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )
    normalizer = BasicTextNormalizer()

    predictions = []
    references = []
    normalized_predictions = []
    normalized_references = []

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to("cuda"),
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(
                    labels != -100, labels, processor.tokenizer.pad_token_id
                )
                decoded_preds = processor.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = processor.tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )
                predictions.extend(decoded_preds)
                references.extend(decoded_labels)
                normalized_predictions.extend(
                    [normalizer(pred).strip() for pred in decoded_preds]
                )
                normalized_references.extend(
                    [normalizer(label).strip() for label in decoded_labels]
                )
            del generated_tokens, labels, batch
        gc.collect()
    wer = 100 * metric.compute(predictions=predictions, references=references)
    normalized_wer = 100 * metric.compute(
        predictions=normalized_predictions, references=normalized_references
    )
    eval_metrics = {"eval/wer": wer, "eval/normalized_wer": normalized_wer}

    print(f"{wer=} and {normalized_wer=}")
    print(eval_metrics)
    return eval_metrics


def generate_w_peft(model, gr_audio, processor, feature_extractor, language):
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )

    audio = Audio(sampling_rate=16000).encode_example(gr_audio)
    audio_input = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features

    model.eval()
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=audio_input.to("cuda"),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=255,
                )
                .cpu()
                .numpy()
            )
    decoded_preds = processor.tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True
    )
    return decoded_preds
