import argparse

from detector import detect, get_prompts
from utils import _load_texts_from_json


def main(
    dataset_name,
    ai_source,
    is_prompt,
    token_size,
    perturbation_num,
    fast_sample_num,
    pct_words_masked,
    ai_source_est,
    self_prompt,
    flag_dict,
):

    model_name = "gpt2-xl"

    additional_string = ""

    file_human = f"{dataset_name}_human"
    file_ai = f"{dataset_name}_from_ai_{ai_source}"

    file_path_human = f"./txtdata/{file_human}.json"

    # file_path_human = f"./txtdata/{file_human}.txt"
    if dataset_name == "HC3":
        file_path_ai = "./txtdata/HC3_ai.json"
    else:
        file_path_ai = f"./txtdata/{file_ai}_{token_size}.json"

    # est_prompt = "est_prompt2"
    # file_prompt_human = (
    #     f"./txtdata/{dataset_name}_{est_prompt}_from_human_{ai_source_est}.json"
    # )
    # file_prompt_ai = (
    #     f"./txtdata/{dataset_name}_{est_prompt}_from_ai_{ai_source_est}.json"
    # )

    prompt_list = list()
    if is_prompt:
        if self_prompt:
            human_texts, ai_texts = (
                _load_texts_from_json(file_path_human),
                _load_texts_from_json(file_path_ai),
            )

            ai_datasize = min(200, len(ai_texts))
            human_datasize = min(200, len(human_texts))

            ai_texts = ai_texts[:ai_datasize]
            human_texts = human_texts[:human_datasize]
            prompt_list = ai_texts + human_texts
        else:
            prompt_list = get_prompts(dataset_name)

    prefix = f"_with_prompt" if is_prompt else ""
    if self_prompt:
        prefix += "_self_prompt"
    # prefix += "fast0.2"

    output_path = f"./results/output{prefix}_{file_ai}_pn{perturbation_num}_fpn{fast_sample_num}_pwm_{pct_words_masked}_binoulars_{flag_dict['is_binoculars']}"
    figure_output_path = f"./plots/roc_curves{prefix}_{file_ai}_pn{perturbation_num}_fpn{fast_sample_num}_pwm_{pct_words_masked}_binoculars_{flag_dict['is_binoculars']}"

    detect(
        file_path_human=file_path_human,
        file_path_ai=file_path_ai,
        output_path=output_path,
        figure_output_path=figure_output_path,
        model_name=model_name,
        fast_sample_num=fast_sample_num,
        perturbation_num=perturbation_num,
        pct_words_masked=pct_words_masked,
        is_prompt=is_prompt,
        prompt_list=prompt_list,
        flag_dict=flag_dict,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai_source", default="llama2")
    parser.add_argument("--dataset_name", default="xsum")
    parser.add_argument("--prompt", action="store_true")
    parser.add_argument("--self_prompt", action="store_true")
    parser.add_argument("--token_size", default=200)
    parser.add_argument("--perturbation_num", default=5, type=int)
    parser.add_argument("--fast_sample_num", default=100, type=int)
    parser.add_argument("--pct_words_num", default=0.1, type=float)
    parser.add_argument("--ai_source_est", default="phi2")
    parser.add_argument("--binoculars_only", action="store_true")

    args = parser.parse_args()
    flag_dict = dict(
        is_entropy=True,
        is_detectgpt=True,
        is_fastdetectgpt=True,
        is_lrr=True,
        is_npr=True,
        is_fastnpr=True,
        is_roberta_detector=True,
        is_log_p=True,
        is_rank=True,
        is_log_rank=True,
        is_max=True,
        is_binoculars=False,
        is_radar=False,
        is_intrinsicPHD=False,
    )
    if args.binoculars_only:
        flag_dict = dict(map(lambda k: (k, False), flag_dict.keys()))
        flag_dict["is_binoculars"] = True

    main(
        args.dataset_name,
        args.ai_source,
        args.prompt,
        args.token_size,
        args.perturbation_num,
        args.fast_sample_num,
        args.pct_words_num,
        args.ai_source_est,
        args.self_prompt,
        flag_dict,
    )
