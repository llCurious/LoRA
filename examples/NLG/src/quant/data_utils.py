import os
import datasets

def get_dataset(dataset_name, type: str = "validation"):
    if dataset_name == "wikitext":
        data_dir = os.path.join(
            os.path.expanduser("~"), ".cache/huggingface/datasets/wikitext"
        )
        # train_dataset = datasets.load_from_disk(data_dir)["train"]
        # eval_dataset = datasets.load_from_disk(data_dir)["validation"]
        ds = datasets.load_from_disk(data_dir)[type]
        return ds
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")