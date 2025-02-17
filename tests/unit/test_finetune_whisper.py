import pytest

from speech_to_text_finetune.finetune_whisper import get_language_id_from_language_name


def test_get_language_id_from_language_name(language_file_path):
    language_id = get_language_id_from_language_name("Abkhaz", language_file_path)
    assert language_id == "ab"

    language_id = get_language_id_from_language_name("Zulu", language_file_path)
    assert language_id == "zu"


def test_get_language_id_from_language_name_unlisted(language_file_path):
    with pytest.raises(KeyError):
        get_language_id_from_language_name("Unlisted language", language_file_path)


def test_get_language_id_from_language_name_missing_lang_file():
    with pytest.raises(FileNotFoundError):
        get_language_id_from_language_name("Zulu", "lang.json")
