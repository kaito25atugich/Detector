import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import _load_texts

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(dataset_name, method, max_new_tokens):
    file_name = f"./txtdata/{dataset_name}_human_for_sum.txt"

    output_path = f"./txtdata/xsum_summary_from_ai_llama_{method}_{max_new_tokens}.json"

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_auth_token=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = _load_texts(file_name)
    output_list = dict()

    with tqdm(dataset) as pbar:
        for i, batch in enumerate(pbar):
            if method == "summarize":
                input_text = f"""[INST] <<SYS>>
                Would you summarize following sentences, please. 
                <</SYS>>
                {batch}[/INST]"""
            elif method == "generation":
                input_text = batch
            input_text_encoded = tokenizer(input_text, return_tensors="pt").to(DEVICE)
            token_size = (
                len(input_text_encoded.input_ids) if method == "summarize" else 30
            )
            input_ids = input_text_encoded.input_ids[:token_size]
            attention_mask = input_text_encoded.attention_mask[:token_size]

            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=40,
                    repetition_penalty=1.0,
                )

            output_text = tokenizer.decode(
                output.sequences[0][len(input_text_encoded.input_ids[0]) :],
                skip_special_tokens=True,
            )
            output_list[i] = output_text

        with open(output_path, "w") as f:
            json.dump(output_list, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="summarize")  # summarize or generation
    parser.add_argument("--max_new_tokens", default=300)
    parser.add_argument("--dataset_name", default="xsum")
    args = parser.parse_args()
    main(args.dataset_name, args.method, args.max_new_tokens)
