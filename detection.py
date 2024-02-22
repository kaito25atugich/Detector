import argparse

from detector import detect, get_prompt_estimation, get_prompt_sum


def main(
    dataset_name,
    ai_source,
    is_prompt,
    is_estimated_prompt,
    token_size,
    perturbation_num,
    fast_sample_num,
    pct_words_masked,
    ai_source_est,
):

    model_name = "gpt2-xl"

    method = "summarize"

    additional_string = ""

    file_human = f"{dataset_name}_human_for_sum"
    file_ai = f"{dataset_name}_summary_from_ai_llama_{method}"

    file_path_human = f"./txtdata/{file_human}.txt"
    file_path_ai = f"./txtdata/{file_ai}.txt"
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
            prompt_list = get_prompt_sum()

    is_entropy = True
    is_detectgpt = True
    is_fastdetectgpt = True
    is_lrr = True
    is_npr = True
    is_fastnpr = True
    is_roberta = True
    is_logp = True
    is_rank = True
    is_log_rank = True
    is_max = True
    is_binoculars = False

    is_radar = False
    is_intrinsicPHD = False

    prefix = (
        f"_with_prompt_est_{bool(is_estimated_prompt)}_{ai_source_est}"
        if is_prompt
        else ""
    )
    # prefix += "fast0.2"

    output_path = f"./results/output{prefix}_{file_ai}_pn{perturbation_num}_fpn{fast_sample_num}_pwm_{pct_words_masked}"
    figure_output_path = f"./plots/roc_curves{prefix}_{file_ai}_pn{perturbation_num}_fpn{fast_sample_num}_pwm_{pct_words_masked}"

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
        fast_sample_num=fast_sample_num,
        perturbation_num=perturbation_num,
        pct_words_masked=pct_words_masked,
        is_prompt=is_prompt,
        prompt_list=prompt_list,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai_source", default="llama2")
    parser.add_argument("--dataset_name", default="xsum")
    parser.add_argument("--prompt", action="store_true")
    parser.add_argument("--estimated_prompt", action="store_true")
    parser.add_argument("--token_size", default=200)
    parser.add_argument("--perturbation_num", default=5, type=int)
    parser.add_argument("--fast_sample_num", default=100, type=int)
    parser.add_argument("--pct_words_num", default=0.1, type=float)
    parser.add_argument("--ai_source_est", default="neural-chat")

    args = parser.parse_args()

    main(
        args.dataset_name,
        args.ai_source,
        args.prompt,
        args.estimated_prompt,
        args.token_size,
        args.perturbation_num,
        args.fast_sample_num,
        args.pct_words_num,
        args.ai_source_est,
    )
