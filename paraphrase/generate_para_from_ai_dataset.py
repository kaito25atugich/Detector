import argparse
import json

from tqdm import tqdm

from paraphraser import (
    PARAPHRASING_ERROR,
    IllegalNonEthicsSentenceError,
    TooShortSentenceError,
    get_paraphraser,
)
from utils import _load_texts, _load_texts_from_json


def main(
    time_num=1,
    method="enhance",
    para_model="llama2",
    file_name="para_from_ai",
    dataset_name="xsum",
):
    datasize = 200
    paraphraser = get_paraphraser(para_model, method)
    para_txts = list()
    para_txts_json = dict()
    para_output_json = dict()

    file_path_ai = (
        f"./txtdata/{dataset_name}_{file_name}_{para_model}_{time_num}time.json"
    )
    ai_texts = _load_texts_from_json(file_path_ai)[:datasize]

    with tqdm(ai_texts) as pbar:
        for i, batch in enumerate(pbar):
            pbar.set_description(f"Creating datasets... ({i}/{datasize})")

            try:
                para_txt, para_txt_all = paraphraser.rephrase(batch)
                para_txts.append(para_txt)
            except (IllegalNonEthicsSentenceError, TooShortSentenceError) as e:
                if isinstance(e, IllegalNonEthicsSentenceError):
                    error = "Unethics"
                elif isinstance(e, TooShortSentenceError):
                    error = "TooShort"
                else:
                    error = "Unknown"
                para_txt = f"{PARAPHRASING_ERROR}:{error}"
                para_txt_all = para_txt
            para_output_json[i] = {
                "input": batch,
                "output": para_txt,
                "output_all": para_txt_all,
            }
            para_txts_json[i] = para_txt

    with open(
        f"./txtdata/{dataset_name}_{file_name}_{para_model}_{time_num+1}time.json", "w"
    ) as f:
        json.dump(para_txts_json, f, ensure_ascii=False)

    with open(
        f"./results/modi_entropy/paraphrase_output_{para_model}_{dataset_name}_{file_name}_{time_num+1}time_jsonver.json",
        "w",
    ) as f:
        json.dump(para_output_json, f, ensure_ascii=False)


if __name__ == "__main__":
    """
    method="enhance",
    para_model="llama2",
    max_token_size=200,
    file_name="para_from_ai",
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", default=0)
    parser.add_argument(
        "--method", default="enhance_with_chatgpt"
    )  # "degrade" or "enhance"
    parser.add_argument("--para_model", default="llama2")
    parser.add_argument("--max_token_size", default=200)
    parser.add_argument(
        "--file_name", default="para_from_ai_dollyv2_7b_doc_enhance_w_chatgpt"
    )
    parser.add_argument("--dataset_name", default="xsum")
    args = parser.parse_args()
    time = int(args.time)
    max_token_size = int(args.max_token_size)
    main(
        time_num=time,
        method=args.method,
        para_model=args.para_model,
        file_name=args.file_name,
        dataset_name=args.dataset_name,
    )
