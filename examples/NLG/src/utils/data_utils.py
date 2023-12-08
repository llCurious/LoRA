import os
import datasets

def get_dataset(dataset_name, type: str = "validation"):
    if dataset_name == "wikitext":
        data_dir = os.path.join(
            os.path.expanduser("~"), ".cache/huggingface/datasets/wikitext"
        )
        dataset = datasets.load_from_disk(data_dir)[type]
    elif dataset_name == "lambada":
        data_dir = os.path.join(
            os.path.expanduser("~"), 
            ".cache/huggingface/datasets/lambada"
        )
        dataset = datasets.load_dataset(data_dir, split=type)
    elif dataset_name in ["cola", "rte", "qnli", "mnli"]:
        data_dir = os.path.join(
            os.path.expanduser("~"), 
            ".cache/huggingface/datasets/glue"
        )
        dataset = datasets.load_dataset("glue", dataset_name, cache_dir=data_dir, split=type)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

    return dataset

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "lambada": ("text", None),
    "wikitext": ("text", None),
}

default_params = {
    "cola": {"num_train_epochs": 50, "max_seq_length": 128},
    "mnli": {"num_train_epochs": 5, "max_seq_length": 128},
    "mrpc": {"num_train_epochs": 20, "max_seq_length": 128},
    "sst2": {"num_train_epochs": 10, "max_seq_length": 128},
    "stsb": {"num_train_epochs": 100, "max_seq_length": 128},
    "qqp": {"num_train_epochs": 5, "max_seq_length": 128},
    "qnli": {"num_train_epochs": 10, "max_seq_length": 128},
    "rte": {"num_train_epochs": 50, "max_seq_length": 128},
    "imdb": {"num_train_epochs": 30, "max_seq_length": 512},
    "wikitext": {"num_train_epochs": 1, "max_seq_length": 50},
    "lambada": {"num_train_epochs": 1, "max_seq_length": 512},
}