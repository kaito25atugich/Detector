import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import _load_texts_from_json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(dataset_name, max_new_tokens):
    file_name = f"./txtdata/{dataset_name}_prompt.json"

    output_path = f"./txtdata/{dataset_name}_from_ai_llama2_{max_new_tokens}.json"

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_auth_token=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = _load_texts_from_json(file_name)
    output_list = dict()

    with tqdm(dataset) as pbar:

        def get_instruct_prompt_llama2(sys_prompt, batch):
            return f"""[INST] <<SYS>>
                {sys_prompt}
                <</SYS>>
                {batch}[/INST]"""

        for i, batch in enumerate(pbar):
            if dataset_name == "xsum":
                sys_prompt = "Would you summarize following sentences, please."
            elif dataset_name == "writingprompts":
                sys_prompt = "Please continue the stropy in the following sentences."
            input_text = get_instruct_prompt_llama2(sys_prompt, batch)
            input_text_encoded = tokenizer(input_text, return_tensors="pt").to(DEVICE)
            input_ids = input_text_encoded.input_ids
            attention_mask = input_text_encoded.attention_mask

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
    parser.add_argument("--max_new_tokens", default=200)
    parser.add_argument("--dataset_name", default="xsum")
    args = parser.parse_args()
    main(args.dataset_name, args.max_new_tokens)
