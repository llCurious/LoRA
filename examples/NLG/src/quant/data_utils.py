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