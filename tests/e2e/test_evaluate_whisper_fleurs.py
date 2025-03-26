import shutil
from pathlib import Path
from speech_to_text_finetune.evaluate_whisper_fleurs import evaluate_fleurs


def test_evaluate_fleurs_e2e():
    results = evaluate_fleurs(
        model_id="openai/whisper-tiny",
        lang_code="af_za",
        language="Afrikaans",
        eval_batch_size=16,
        n_test_samples=10,
        fp16=False,
    )

    expected_dir_path = Path("artifacts/af_za_google_fleurs")
    assert expected_dir_path.exists()

    assert 5.16 < results["eval_loss"] < 5.18
    assert 82.24 < results["eval_wer"] < 82.26
    assert 84.83 < results["eval_wer_ortho"] < 84.85
    assert 34.44 < results["eval_cer"] < 34.46
    assert 36.82 < results["eval_cer_ortho"] < 36.84

    shutil.rmtree("artifacts")
