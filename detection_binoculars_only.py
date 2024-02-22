import argparse

from detector import detect, get_prompt_estimation, get_prompt_sum


def main(dataset_name, ai_source, is_prompt, is_estimated_prompt, ai_source_est):

    model_name = "gpt2-xl"

    method = "summarize"

    token_size = 200

    additional_string = ""

    file_human = f"{dataset_name}_human_for_sum"
    file_ai = f"{dataset_name}_summary_from_ai_llama_{method}"

    file_path_human = f"./txtdata/{file_human}.txt"
    file_path_ai = f"./txtdata/{file_ai}.json"

    file_prompt_human = (
        f"./txtdata/xsum_est_prompt_for_summary_from_human_{ai_source_est}.json"
    )
    file_prompt_ai = (
        f"./txtdata/xsum_est_prompt_for_summary_from_ai_{ai_source_est}.json"
    )

    prompt_list = list()
    if is_prompt:
        if is_estimated_prompt:
            prompt_list += get_prompt_estimation(file_prompt_human)
            prompt_list += get_prompt_estimation(file_prompt_ai)
        else:
            prompt_list += get_prompt_sum()

    is_entropy = False
    is_detectgpt = False
    is_fastdetectgpt = False
    is_lrr = False
    is_npr = False
    is_fastnpr = False
    is_roberta = False
    is_logp = False
    is_rank = False
    is_log_rank = False
    is_max = False
    is_intrinsicPHD = False
    is_radar = False
    is_binoculars = True

    prefix = (
        f"_with_prompt_est_{bool(is_estimated_prompt)}_{ai_source_est}"
        if is_prompt
        else ""
    )
    prefix += "_only_binoculars"

    output_path = f"./results/output{prefix}_{file_ai}"
    figure_output_path = f"./plots/roc_curves{prefix}_{file_ai}"

    detect(
        file_path_human=file_path_human,
        file_path_ai=file_path_ai,
        output_path=output_path,
        figure_output_path=figure_output_path,
        model_name=model_name,
        is_entropy=is_entropy,
        is_detectgpt=is_detectgpt,
        is_fastdetectgpt=is_fastdetectgpt,
        is_lrr=is_lrr,
        is_npr=is_npr,
        is_fastnpr=is_fastnpr,
        is_roberta=is_roberta,
        is_rank=is_rank,
        is_log_rank=is_log_rank,
        is_logp=is_logp,
        is_max=is_max,
        is_intrinsicPHD=is_intrinsicPHD,
        is_radar=is_radar,
        is_binoculars=is_binoculars,
        is_prompt=is_prompt,
        is_estimation_prompt=is_estimated_prompt,
        prompt_list=prompt_list,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai_source", default="llama")
    parser.add_argument("--ai_source_est", default="phi-2")
    parser.add_argument("--dataset_name", default="xsum")
    parser.add_argument("--prompt", action="store_true")
    parser.add_argument("--estimated_prompt", action="store_true")

    args = parser.parse_args()

    main(
        args.dataset_name,
        args.ai_source,
        args.prompt,
        args.estimated_prompt,
        args.ai_source_est,
    )
