import json

import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from transformers import AutoTokenizer

# 15 colorblind-friendly colors
COLORS = [
    "#0072B2",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#000000",
    "#0072B2",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
]


def _load_texts(file_path):
    with open(file_path, "r") as f:
        data = f.read()
        data_list = data.split("\n")

    return data_list


def _load_texts_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    data_list = [d for d in data.values() if d.replace("\n", "") != ""]
    return data_list


def find_max_index(target_list: list[float], threshold: float):
    max_index = -1
    for i in range(len(target_list)):
        if target_list[i] <= threshold:
            max_index = i
    return max_index


def get_roc_metrics(true_labels, scores):
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    thresholds_dict = dict()
    for fpr_rate in [0.01, 0.05, 0.1, 0.25]:
        idx = find_max_index(tpr, fpr_rate)
        thresholds_dict[fpr_rate] = thresholds[idx]

    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc), thresholds_dict


def get_precision_recall_metrics(true_labels, scores):
    precision, recall, _ = precision_recall_curve(true_labels, scores)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)


def get_output_json(name, fpr, tpr, roc_auc, true_label, prediction_score, thresholds):
    return {
        "name": name,
        "true_label": true_label,
        "prediction_score": prediction_score,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "threshold": thresholds,
    }


def save_roc_curves(output_file, experiments, model_name, mask_model="t5_large"):
    # first, clear plt
    plt.clf()

    for experiment, color in zip(experiments.values(), COLORS):
        metrics = experiment
        plt.plot(
            metrics["fpr"],
            metrics["tpr"],
            label=f"{experiment['name']}, roc_auc={metrics['roc_auc']:.3f}",
            color=color,
        )
        # print roc_auc for this experiment
        print(f"{experiment['name']} roc_auc: {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves ({model_name} - {mask_model})")
    plt.legend(loc="lower right", fontsize=6)
    plt.savefig(output_file)


def add_result_to_output_json(name, label_list, scores, output_json):
    fpr, tpr, roc_auc, thresholds = get_roc_metrics(label_list, scores)
    output_json[name] = get_output_json(
        name, fpr, tpr, roc_auc, label_list, scores, thresholds
    )

    return output_json


def assert_tokenizer_consistency(model_id_1, model_id_2):
    identical_tokenizers = (
        AutoTokenizer.from_pretrained(model_id_1, token=True).vocab
        == AutoTokenizer.from_pretrained(model_id_2, token=True).vocab
    )
    if not identical_tokenizers:
        raise ValueError(
            f"Tokenizers are not identical for {model_id_1} and {model_id_2}."
        )
