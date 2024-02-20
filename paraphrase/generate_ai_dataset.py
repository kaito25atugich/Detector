import json

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_texts(file_path):
    with open(file_path, "r") as f:
        data = f.read()
        data_list = data.split("\n")

    return data_list


def main():
    dataset_name = "xsum"
    model_name = "databricks/dolly-v2-7b"  # instrusted fine-tune
    # model_name = "Writer/palmyra-base"  # instructed fine-tune
    datasize = 200
    ai_txts = dict()
    max_new_tokens = 300

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    file_path_human = f"./txtdata/{dataset_name}_human_doc.txt"
    human_texts = _load_texts(file_path_human)
    counter = 0
    with tqdm(human_texts) as pbar:
        for i, batch in enumerate(pbar):
            pbar.set_description(f"Creating datasets... ({counter}/{datasize})")
            if counter >= datasize:
                break
            input_text = tokenizer(batch)
            input_text_truncated = dict(
                input_ids=torch.tensor([input_text["input_ids"][:30]]).to(DEVICE),
                attention_mask=torch.tensor([input_text["attention_mask"][:30]]).to(
                    DEVICE
                ),
            )
            output_txt = model.generate(
                **input_text_truncated,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.95,
                top_k=40,
                repetition_penalty=1.0,
            )
            output_txt_decoded = tokenizer.decode(
                output_txt[0], skip_special_tokens=True
            )
            ai_txts[i] = output_txt_decoded

    with open(
        f"./txtdata/{dataset_name}_para_from_ai_dollyv2_7b_doc_pena1_llama2_0time.json",
        "w",
    ) as f:
        json.dump(ai_txts, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
