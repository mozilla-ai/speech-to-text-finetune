import yaml
from pydantic import BaseModel


def load_config(config_path: str):
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    return Config(**config_dict)


class TrainingConfig(BaseModel):
    """
    More info at https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments
    """

    push_to_hub: bool
    hub_private_repo: bool
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    gradient_checkpointing: bool
    fp16: bool
    eval_strategy: str
    per_device_eval_batch_size: int
    predict_with_generate: bool
    generation_max_length: int
    save_steps: int
    logging_steps: int
    load_best_model_at_end: bool
    save_total_limit: int
    metric_for_best_model: str
    greater_is_better: bool


class Config(BaseModel):
    """
    Store configuration used for finetuning

    Args:
        model_id (str): HF model id of a Whisper model used for finetuning
        dataset_id (str): HF dataset id of a Common Voice dataset version, ideally from the mozilla-foundation repo
        dataset_source (str): can be "HF" or "local", to determine from where to fetch the dataset
        language (str): registered language string that is supported by the Common Voice dataset
        repo_name (str): used both for local dir and HF, "default" will create a name based on the model and language id
        training_hp (TrainingConfig): store selective hyperparameter values from Seq2SeqTrainingArguments
    """

    model_id: str
    dataset_id: str
    dataset_source: str
    language: str
    repo_name: str
    training_hp: TrainingConfig


LANGUAGES_NAME_TO_ID = {
    "Abkhaz": "ab",
    "Acehnese": "ace",
    "Adyghe": "ady",
    "Afrikaans": "af",
    "Amharic": "am",
    "Aragonese": "an",
    "Arabic": "ar",
    "Mapudungun": "arn",
    "Assamese": "as",
    "Asturian": "ast",
    "Azerbaijani": "az",
    "Bashkir": "ba",
    "Basaa": "bas",
    "Belarusian": "be",
    "Bulgarian": "bg",
    "Bambara": "bm",
    "Bengali": "bn",
    "Tibetan": "bo",
    "Breton": "br",
    "Bosnian": "bs",
    "Buryat": "bxr",
    "Medumba": "byv",
    "Catalan": "ca",
    "Kaqchikel": "cak",
    "Central Kurdish": "ckb",
    "Hakha Chin": "cnh",
    "Corsican": "co",
    "Crimean Tatar": "crh",
    "Czech": "cs",
    "Chuvash": "cv",
    "Welsh": "cy",
    "Danish": "da",
    "Dagbani": "dag",
    "German": "de",
    "Sorbian, Lower": "dsb",
    "Dhivehi": "dv",
    "Dioula": "dyu",
    "Greek": "el",
    "English": "en",
    "Esperanto": "eo",
    "Spanish": "es",
    "Estonian": "et",
    "Basque": "eu",
    "Ewondo": "ewo",
    "Persian": "fa",
    "Fulah": "ff",
    "Finnish": "fi",
    "Faroese": "fo",  # codespell:ignore
    "French": "fr",
    "Pular Guinea": "fuf",
    "Frisian": "fy-NL",
    "Irish": "ga-IE",
    "Galician": "gl",
    "Guarani": "gn",
    "Goan Konkani": "gom",
    "Gujarati": "gu-IN",
    "Wayuunaiki": "guc",
    "Hausa": "ha",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hiligaynon": "hil",
    "Croatian": "hr",
    "Sorbian, Upper": "hsb",
    "Haitian": "ht",
    "Hungarian": "hu",
    "Armenian": "hy-AM",
    "Armenian Western": "hyw",
    "Interlingua": "ia",
    "Indonesian": "id",
    "Interlingue": "ie",
    "Igbo": "ig",
    "Icelandic": "is",
    "Italian": "it",
    "Izhorian": "izh",
    "Japanese": "ja",
    "Lojban": "jbo",
    "Javanese": "jv",
    "Georgian": "ka",
    "Karakalpak": "kaa",
    "Kabyle": "kab",
    "Kabardian": "kbd",
    "Kikuyu": "ki",
    "Kazakh": "kk",
    "Khmer": "km",
    "Kurmanji Kurdish": "kmr",
    "Kannada": "kn",
    "Konkani (Devanagari)": "knn",
    "Korean": "ko",
    "Komi-Zyrian": "kpv",
    "Cornish": "kw",
    "Kyrgyz": "ky",
    "Luxembourgish": "lb",
    "Luganda": "lg",
    "Ligurian": "lij",
    "Lingala": "ln",
    "Lao": "lo",
    "Lithuanian": "lt",
    "Latgalian": "ltg",
    "Latvian": "lv",
    "Laz": "lzz",
    "Maithili": "mai",
    "Moksha": "mdf",
    "Malagasy": "mg",
    "Meadow Mari": "mhr",
    "Macedonian": "mk",
    "Malayalam": "ml",
    "Mongolian": "mn",
    "Meetei Lon": "mni",
    "Mossi": "mos",
    "Marathi": "mr",
    "Hill Mari": "mrj",
    "Malay": "ms",
    "Maltese": "mt",
    "Burmese": "my",
    "Erzya": "myv",
    "Taiwanese (Minnan)": "nan-tw",
    "Norwegian Bokm\u00e5l": "nb-NO",
    "IsiNdebele (North)": "nd",  # codespell:ignore
    "Nepali": "ne-NP",
    "Eastern Huasteca Nahuatl": "nhe",
    "Western Sierra Puebla Nahuatl": "nhi",
    "Nias": "nia",
    "Dutch": "nl",
    "Norwegian Nynorsk": "nn-NO",
    "IsiNdebele (South)": "nr",
    "Northern Sotho": "nso",
    "Chinyanja": "ny",
    "Runyankole": "nyn",
    "Occitan": "oc",
    "Afaan Oromo": "om",
    "Odia": "or",
    "Ossetian": "os",
    "Punjabi": "pa-IN",
    "Papiamento (Aruba)": "pap-AW",
    "Polish": "pl",
    "Pashto": "ps",
    "Portuguese": "pt",
    "K'iche'": "quc",
    "Quechua Chanka": "quy",
    "Kichwa": "qvi",
    "Romansh Sursilvan": "rm-sursilv",
    "Romansh Vallader": "rm-vallader",
    "Romanian": "ro",
    "Russian": "ru",
    "Kinyarwanda": "rw",
    "Sakha": "sah",
    "Santali (Ol Chiki)": "sat",
    "Sardinian": "sc",
    "Sicilian": "scn",
    "Scots": "sco",
    "Sindhi": "sd",
    "Southern Kurdish": "sdh",
    "Shilha": "shi",
    "Sinhala": "si",
    "Slovak": "sk",
    "Saraiki": "skr",
    "Slovenian": "sl",
    "Soninke": "snk",
    "Somali": "so",
    "Albanian": "sq",
    "Serbian": "sr",
    "Siswati": "ss",
    "Southern Sotho": "st",
    "Swedish": "sv-SE",
    "Swahili": "sw",
    "Syriac": "syr",
    "Tamil": "ta",
    "Telugu": "te",  # codespell:ignore
    "Tajik": "tg",
    "Thai": "th",
    "Tigrinya": "ti",
    "Tigre": "tig",
    "Turkmen": "tk",
    "Tagalog": "tl",
    "Setswana": "tn",
    "Toki Pona": "tok",
    "Turkish": "tr",
    "Xitsonga": "ts",
    "Tatar": "tt",
    "Twi": "tw",
    "Tahitian": "ty",
    "Tuvan": "tyv",
    "Ubykh": "uby",
    "Udmurt": "udm",
    "Uyghur": "ug",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Tshivenda": "ve",
    "Venetian": "vec",
    "Vietnamese": "vi",
    "Emakhuwa": "vmw",
    "Votic": "vot",
    "Westphalian": "wep",
    "Wolof": "wo",
    "Xhosa": "xh",
    "Yiddish": "yi",
    "Yoruba": "yo",
    "Cantonese": "yue",
    "Tamazight": "zgh",
    "Chinese (China)": "zh-CN",
    "Chinese (Hong Kong)": "zh-HK",
    "Chinese (Taiwan)": "zh-TW",
    "Zulu": "zu",
    "Zaza": "zza",
}
