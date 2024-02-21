import argparse
import json
import random

import numpy as np
from evaluate import load
from transformers import logging

logging.set_verbosity_error()


def main():
    dataset_name = "xsum"
    file_a_path = "./txtdata/xsum_est_prompt_for_summary_from_ai_llama.json"
    file_b_path = "./txtdata/xsum_est_prompt_for_summary_from_human.json"

    with open(file_a_path, "r") as f:
        ai_prompt_texts = json.load(f)

    with open(file_b_path, "r") as f:
        human_prompt_texts = json.load(f)

    size = 10

    ai_texts = random.sample(list(ai_prompt_texts.values()), size)
    human_texts = random.sample(list(human_prompt_texts.values()), size)

    bertscore = load("bertscore")

    results = bertscore.compute(predictions=ai_texts, references=human_texts, lang="en")

    print(
        f"BERT SCORE (mean): P{np.mean(results['f1'])}, num_of_sample: {len(results['f1'])}"
    )


if __name__ == "__main__":
    main()
