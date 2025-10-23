# 🎨 **Customization Guide**

This Blueprint is designed to be flexible and easily adaptable to your specific needs. This guide will walk you through some key areas you can customize to make the Blueprint your own.

---

## BYOD: Bring Your Own Dataset

> But I already have my own speech-text dateset! I don't want to create a new one from scratch or use Common Voice!
> Does this Blueprint have anything to offer me?

**But of course!**

This guide will walk you through how to use the existing codebase to adapt it to your own unique dataset with minimal effort.

The idea is to load and pre-process your own dataset in the same format as the existing datasets, allowing you to seamlessly integrate with the `finetune_whisper.py` script.

### Step 1: Understand your Dataset

Before creating your custom dataset loading function, it's essential to understand the data format that the `finetune_whisper.py` script expects. Typically, the dataset should have a structure that looks a bit like this:

```python
{
    "train": [
        {
            "audio": "path/to/audio_file.wav",
            "text": "The transcribed text of the audio"
        },
        # More examples...
    ],
    "test": [
        {
            "audio": "path/to/audio_file_2.wav",
            "text": "Another transcribed text"
        },
        # More examples...
    ]
}
```

Notably, there should be a pair of transcribed text and an audio clip (usually in the form of a path to the audio file, either `.mp3` or `.wav`)

### Step 2: Create a *load_my_dataset* Function

Next, you'll create a custom dataset loading function and place it inside `data_process.py`. This function will load and pre-process your dataset into the expected format and return an HuggingFace `DatasetDict` containing two `Dataset` objects like so: `train`:`Dataset` and `test`:`Dataset` for each split respectively.

As an example, lets consider that you have a directory with a csv file and all the audio clips like this:

```
datasets/
├── my_dataset/
│   ├── dataset.csv
│   └── audio_files/
│       ├── audio_1.wav
│       ├── audio_2.wav
│       ├── audio_3.wav
│       └── ...
```
and that the .csv file has the following format:

```
csv my_dataset/dataset.csv
audio,text
example_1.mp3,"This is an example"
example_2.mp3,"This is another example"
...
example_n.mp3,"This is yet another example"
```

**Example function**
```
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def load_my_dataset(my_dataset_dir: str = "/home/user/datasets/my_dataset", train_split: float = 0.8) -> DatasetDict:
    """
    Load and process your custom dataset from the given directory.

    Args:
        my_dataset_dir (str): Path to the directory containing your dataset.
        train_split (float): Percentage of data to use for training. The rest will be used for evaluation as a test set.

    Returns:
        DatasetDict: HF Dataset dictionary that consists of two distinct Datasets (train and test)
    """
    # Define the path to the CSV file
    csv_path = os.path.join(my_dataset_dir, "dataset.csv")
    df = pd.read_csv(csv_path)

    # Only keep the columns we need and drop any other possible metadata columns
    df = df.select_columns(["audio", "sentence"])

    # Our processing script expects the column with the transcribed text to be called "sentence"
    df = df.rename(columns={"text": "sentence"})

    # Replace the relative path to the audio clip with the absolute path
    df["audio"] = df["audio"].apply(lambda p: f"{my_dataset_dir}/audio_files/{p}")

    # Split the DataFrame into train and test sets and set a seed to shuffle and easily reproduce
    train_df, test_df = train_test_split(df, train_size=train_split, random_state=42)

    # Return the DatasetDict containing the train and test datasets
    return DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df)
    })

```

### Step 3: Integrate with the rest of the codebase

Once you've created your custom dataset loading function, you need to integrate it into the existing codebase. Specifically, you need to update the `load_dataset_from_dataset_id` function in `data_process.py` to include the new function. Simply add to the function:


```
    try:
        dataset = load_my_dataset(dataset_id)
        return dataset, _get_local_proc_dataset_path(dataset_id)
    except FileNotFoundError:
        pass
```

### Step 4: Update your config file

Don't forget to update your config file so that the `dataset_id` points to the right directory path:

```
model_id: openai/whisper-tiny
dataset_id: /home/user/datasets/my_dataset
...
```


### Step 5: Fine-Tune the model with your own dataset!

Finally, simply run the finetune_whisper.py script to fine-tune the Whisper model using your custom dataset.

```
python src/speech_to_text_finetune/finetune_whisper.py
```


## 🤝 **Contributing to the Blueprint**

Want to help improve or extend this Blueprint? Check out the **[Future Features & Contributions Guide](future-features-contributions.md)** to see how you can contribute your ideas, code, or feedback to make this Blueprint even better!
