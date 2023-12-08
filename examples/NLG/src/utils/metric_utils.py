from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from torch.nn import CrossEntropyLoss
import torch

def simple_accuracy(preds, labels):
    preds = preds.numpy()
    labels = labels.numpy()
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def simple_ppl(preds, labels):
    # NOTE: older version, it seems to calculate the wrong PPL
    # loss = torch.sum(
    #     torch.nn.functional.log_softmax(preds, dim=-1)
    #     * torch.nn.functional.one_hot(labels, num_classes=preds.shape[-1]),
    # )
    # count = torch.sum(labels != tokenizer.pad_token_id)
    # logging.info(f"loss: {loss}, count: {count}")
    # perplexity = torch.exp(-loss / count)

    # NOTE: reference: https://github.com/huggingface/transformers/issues/473
    shift_logits = preds.contiguous()
    shift_labels = labels.contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    cls_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    perplexity = torch.exp(cls_loss)
    return perplexity

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "stsb":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "imdb":
        return {"imdb": simple_accuracy(preds, labels)}
    elif task_name == "wikitext":
        return {"ppl": simple_ppl(preds, labels)}
    elif task_name == "lambada":
        return {"ppl": simple_ppl(preds, labels)}
    else:
        raise KeyError(task_name)