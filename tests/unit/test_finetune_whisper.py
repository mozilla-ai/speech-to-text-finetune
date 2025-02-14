import pytest

from speech_to_text_finetune.finetune_whisper import get_language_id_from_language_name

default_lang_path = "artifacts/languages_common_voice_17_0.json"


def test_get_language_id_from_language_name():
    language_id = get_language_id_from_language_name("Abkhaz", default_lang_path)
    assert language_id == "ab"

    language_id = get_language_id_from_language_name("Zulu", default_lang_path)
    assert language_id == "zu"


def test_get_language_id_from_language_name_unlisted():
    with pytest.raises(KeyError):
        get_language_id_from_language_name("Unlisted language", default_lang_path)


def test_get_language_id_from_language_name_missing_lang_file():
    with pytest.raises(FileNotFoundError):
        get_language_id_from_language_name("Zulu", "lang.json")
