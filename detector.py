import json
import re
from threading import Thread
from typing import Union

import numpy as np
import torch
import transformers
from scipy.spatial.distance import cdist
from torch.nn import functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from utils import (
    _load_texts,
    _load_texts_from_json,
    add_result_to_output_json,
    assert_tokenizer_consistency,
    save_roc_curves,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# For DetectGPT
# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")

HUMAN_GENERATED = 0
AI_GENERATED = 1


MIN_SUBSAMPLE = 20
INTERMEDIATE_POINTS = 4


def prim_tree(adj_matrix, alpha=1.0):
    infty = np.max(adj_matrix) + 10

    dst = np.ones(adj_matrix.shape[0]) * infty
    visited = np.zeros(adj_matrix.shape[0], dtype=bool)
    ancestor = -np.ones(adj_matrix.shape[0], dtype=int)

    v, s = 0, 0.0
    for i in range(adj_matrix.shape[0] - 1):
        visited[v] = 1
        ancestor[dst > adj_matrix[v]] = v
        dst = np.minimum(dst, adj_matrix[v])
        dst[visited] = infty

        v = np.argmin(dst)
        s += adj_matrix[v][ancestor[v]] ** alpha

    return s.item()


def preprocess_string(sss):
    return sss.replace("\n", " ").replace("\t", " ").replace("  ", " ")


class PHD:
    def __init__(
        self, alpha=1.0, metric="euclidean", n_reruns=3, n_points=7, n_points_min=3
    ):
        """
        Initializes the instance of PH-dim computer
        Parameters:
            1) alpha --- real-valued parameter Alpha for computing PH-dim (see the reference paper). Alpha should be chosen lower than
        the ground-truth Intrinsic Dimensionality; however, Alpha=1.0 works just fine for our kind of data.
            2) metric --- String or Callable, distance function for the metric space (see documentation for Scipy.cdist)
            3) n_reruns --- Number of restarts of whole calculations (each restart is made in a separate thread)
            4) n_points --- Number of subsamples to be drawn at each subsample
            5) n_points_min --- Number of subsamples to be drawn at larger subsamples (more than half of the point cloud)
        """
        self.alpha = alpha
        self.n_reruns = n_reruns
        self.n_points = n_points
        self.n_points_min = n_points_min
        self.metric = metric
        self.is_fitted_ = False

    def _sample_W(self, W, nSamples):
        n = W.shape[0]
        random_indices = np.random.choice(n, size=nSamples, replace=False)
        return W[random_indices]

    def _calc_ph_dim_single(self, W, test_n, outp, thread_id):
        lengths = []
        for n in test_n:
            if W.shape[0] <= 2 * n:
                restarts = self.n_points_min
            else:
                restarts = self.n_points

            reruns = np.ones(restarts)
            for i in range(restarts):
                tmp = self._sample_W(W, n)
                reruns[i] = prim_tree(cdist(tmp, tmp, metric=self.metric), self.alpha)

            lengths.append(np.median(reruns))
        lengths = np.array(lengths)

        x = np.log(np.array(list(test_n)))
        y = np.log(lengths)
        N = len(x)
        outp[thread_id] = (N * (x * y).sum() - x.sum() * y.sum()) / (
            N * (x**2).sum() - x.sum() ** 2
        )

    def fit_transform(self, X, y=None, min_points=50, max_points=512, point_jump=40):
        """
        Computing the PH-dim
        Parameters:
            1) X --- point cloud of shape (n_points, n_features),
            2) y --- fictional parameter to fit with Sklearn interface
            3) min_points --- size of minimal subsample to be drawn
            4) max_points --- size of maximal subsample to be drawn
            5) point_jump --- step between subsamples
        """
        ms = np.zeros(self.n_reruns)
        test_n = range(min_points, max_points, point_jump)
        threads = []

        for i in range(self.n_reruns):
            # self._calc_ph_dim_single(X, test_n, ms, i)
            threads.append(
                Thread(target=self._calc_ph_dim_single, args=[X, test_n, ms, i])
            )
            threads[-1].start()

        for i in range(self.n_reruns):
            threads[i].join()

        m = np.mean(ms)
        return 1 / (1 - m)


ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)


def perplexity(
    encoding: transformers.BatchEncoding,
    logits: torch.Tensor,
    median: bool = False,
    temperature: float = 1.0,
    prompt_len: int = 0,
):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., prompt_len + 1 :].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., prompt_len + 1 :].contiguous()

    if median:
        ce_nan = ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).masked_fill(
            ~shifted_attention_mask.bool(), float("nan")
        )
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

    else:
        ppl = (
            ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)
            * shifted_attention_mask
        ).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()

    return ppl


def entropy(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    encoding: transformers.BatchEncoding,
    pad_token_id: int,
    median: bool = False,
    sample_p: bool = False,
    temperature: float = 1.0,
    prompt_len: int = 0,
):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(
            p_proba.view(-1, vocab_size), replacement=True, num_samples=1
        ).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids[:, prompt_len:] != pad_token_id).type(
        torch.uint8
    )

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (
            ((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy()
        )

    return agg_ce


class Binoculars(object):
    def __init__(
        self,
        observer_name_or_path: str = "tiiuae/falcon-7b",
        performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
        use_bfloat16: bool = True,
        max_token_observed: int = 512,
        mode: str = "low-fpr",
    ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path,
            device_map={"": DEVICE},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            # torch_dtype=torch.float16,
            use_auth_token=True,
        )
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path,
            device_map={"": DEVICE},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            # torch_dtype=torch.float16,
            use_auth_token=True,
        )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            observer_name_or_path, use_auth_token=True
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        ).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(DEVICE)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE)).logits
        if DEVICE != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(
        self, input_text: Union[list[str], str], prompt
    ) -> Union[float, list[float]]:
        if prompt:
            batch = [prompt + input_text]
        else:
            batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        prompt_len = 0
        if prompt:
            prompt_len = len(self._tokenize(prompt)[0])
            observer_logits = observer_logits[:, prompt_len:]
            performer_logits = performer_logits[:, prompt_len:]
        ppl = perplexity(encodings, performer_logits, prompt_len=prompt_len)
        x_ppl = entropy(
            observer_logits.to(DEVICE),
            performer_logits.to(DEVICE),
            encodings.to(DEVICE),
            self.tokenizer.pad_token_id,
            prompt_len=prompt_len,
        )
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return (
            binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores
        )


class Detector:
    def __init__(
        self,
        model_name: str,
        seed_num: int = 0,
        is_entropy: bool = True,
        ## setting parameter for detectGPT
        is_detectgpt: bool = True,
        mask_filling_model_name="t5-large",
        max_mask_model_size=512,
        num_of_perturbation=5,
        pct_words_masked=0.1,
        span_size=1,
        buffer_size=0,
        mask_top_p=1.0,
        is_fastdetectgpt: bool = True,
        fast_sample_num=10000,
        ##
        is_roberta_detector: bool = True,
        is_log_p: bool = True,
        is_rank: bool = True,
        is_log_rank: bool = True,
        is_max: bool = True,
        is_lrr: bool = True,
        is_npr: bool = True,
        is_fastnpr: bool = True,
        is_intrinsicPHD: bool = True,
        is_radar: bool = True,
        is_binoculars: bool = False,
        is_prompt: bool = False,
        prompt_list: list[str] = [],
        label_list: list[int] = [],
    ):
        self.model_name = model_name
        self.seed_num = seed_num
        self.is_entropy = is_entropy
        self.coefficient_list = list()

        self.fast_sample_num = fast_sample_num
        self.is_fastdetectgpt = is_fastdetectgpt

        self.is_detectgpt = is_detectgpt
        self.mask_filling_model_name = mask_filling_model_name
        self.max_mask_model_size = max_mask_model_size
        self.num_of_perturbation = num_of_perturbation
        self.pct_words_masked = pct_words_masked
        self.span_size = span_size
        self.buffer_size = buffer_size
        self.mask_top_p = mask_top_p

        self.is_roberta_detector = is_roberta_detector
        self.is_log_p = is_log_p
        self.is_rank = is_rank
        self.is_log_rank = is_log_rank

        self.is_max = is_max
        self.is_lrr = is_lrr
        self.is_npr = is_npr
        self.is_fastnpr = is_fastnpr
        self.is_intrinsicPHD = is_intrinsicPHD
        self.is_radar = is_radar
        self.is_binoculars = is_binoculars

        self.is_prompt = is_prompt
        self.prompt_list = prompt_list

        self.label_list = label_list
        self.floating_point = torch.float16

        # lists to store the score
        self.detect_gpt = "detectgpt"
        self.fastdetect_gpt = "fastdetectgpt"
        self.lrr = "lrr"
        self.npr = "npr"
        self.fastnpr = "fastnpr"
        self.entropy = "entropy"
        self.log_p = "logp"
        self.rank = "rank"
        self.log_rank = "log_rank"
        self.max = "max"
        self.roberta_base = "roberta_base"
        self.intrinsicPHD = "intrinsicPHD"
        self.radar = "radar"
        self.binoculars = "binoculars"
        self.detectors = [
            self.detect_gpt,
            self.fastdetect_gpt,
            self.lrr,
            self.npr,
            self.fastnpr,
            self.entropy,
            self.log_p,
            self.rank,
            self.log_rank,
            self.max,
            self.roberta_base,
            self.intrinsicPHD,
            self.radar,
            self.binoculars,
        ]
        self.scores = dict()
        self.set_score_list()

        self.load_model_and_tokenizer()

        if is_detectgpt or is_npr or is_intrinsicPHD:
            self.load_mask_model()

        if is_roberta_detector:
            tokenizer_kwargs = {"truncation": True, "max_length": 512}
            self.roberta_detector = pipeline(
                "text-classification",
                model="roberta-base-openai-detector",
                **tokenizer_kwargs,
            )
        if is_intrinsicPHD:
            mask_model_name = "roberta-large"
            self.encoder_mask_model = AutoModelForMaskedLM.from_pretrained(
                mask_model_name, device_map="auto", torch_dtype=self.floating_point
            )
            self.encoder_mask_tokenizer = AutoTokenizer.from_pretrained(
                mask_model_name, model_max_length=512
            )
            self.encoder_mask_model.eval()
        if is_radar:
            radar_model_name = "TrustSafeAI/RADAR-Vicuna-7B"
            self.radar_model = AutoModelForSequenceClassification.from_pretrained(
                radar_model_name, device_map="auto", torch_dtype=self.floating_point
            )
            self.radar_tokenizer = AutoTokenizer.from_pretrained(
                radar_model_name, model_max_length=512
            )
            self.radar_model.eval()
        if is_binoculars:
            observer_name_or_path: str = "tiiuae/falcon-7b"
            performer_name_or_path: str = "tiiuae/falcon-7b-instruct"

            observer_name_or_path = "01-ai/Yi-6B"
            performer_name_or_path = "HenryJJ/Instruct_Yi-6B_Dolly15K"

            observer_name_or_path = "openlm-research/open_llama_3b_v2"
            performer_name_or_path = "mediocredev/open-llama-3b-v2-instruct"

            self.bino = Binoculars(
                observer_name_or_path=observer_name_or_path,
                performer_name_or_path=performer_name_or_path,
            )

    def load_model_and_tokenizer(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype=self.floating_point
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()

    def load_mask_model(self):
        self.mask_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.mask_filling_model_name,
            torch_dtype=self.floating_point,
            device_map="auto",
        )
        self.mask_tokenizer = AutoTokenizer.from_pretrained(
            self.mask_filling_model_name, model_max_length=self.max_mask_model_size
        )
        self.mask_model.eval()

    def set_score_list(self):
        for method in self.detectors:
            self.scores[method] = 0

    def detect(self, input_text, prompt_label):
        if prompt_label >= 0:
            prompt_token_len = len(
                self.tokenizer.encode(self.prompt_list[prompt_label])
            )
            input_text_with_prompt = self.prompt_list[prompt_label] + input_text
            encoded_inputs = self.tokenizer(
                input_text_with_prompt, return_tensors="pt", truncation=True
            ).to(DEVICE)
            model_output = self.model(
                **encoded_inputs, labels=encoded_inputs["input_ids"]
            )
            self.handle_detector(
                model_output.logits[:, prompt_token_len:-1],
                encoded_inputs.input_ids[:, prompt_token_len + 1 :],
                input_text,
                self.prompt_list[prompt_label],
            )
        else:
            encoded_inputs = self.tokenizer(
                input_text, return_tensors="pt", truncation=True
            ).to(DEVICE)
            model_output = self.model(
                **encoded_inputs, labels=encoded_inputs["input_ids"]
            )
            self.handle_detector(
                model_output.logits[:, :-1],
                encoded_inputs.input_ids[:, 1:],
                input_text,
                "",
            )

    def handle_detector(self, logits, label, input_text, prompt):
        if self.is_detectgpt or self.is_npr:
            torch.manual_seed(self.seed_num)
            np.random.seed(self.seed_num)
            perturbed_texts = [
                self.perturb_text(input_text) for _ in range(self.num_of_perturbation)
            ]
            if prompt:
                perturbed_texts = [prompt + ptxt for ptxt in perturbed_texts]

        if self.is_entropy:
            self.entropy_detection(logits)

        if self.is_detectgpt:
            self.detectgpt_detection(perturbed_texts, logits, label)

        if self.is_fastdetectgpt:
            self.fastdetectgpt_detection(logits, label)

        if self.is_roberta_detector:
            if prompt:
                input_text = prompt + input_text
            self.roberta_detection(input_text)

        if self.is_log_p:
            self.log_p_detection(logits, label)

        if self.is_rank:
            self.rank_detection(logits, label)

        if self.is_log_rank:
            self.log_rank_detection(logits, label)

        if self.is_max:
            self.max_detection(logits)

        if self.is_lrr:
            self.lrr_detection(
                logits,
                label,
            )

        if self.is_npr:
            self.npr_detection(perturbed_texts, logits, label)

        if self.is_fastnpr:
            self.fastnpr_detection(logits, label)

        if self.is_intrinsicPHD:
            if prompt:
                input_text = prompt + input_text
            self.intrinsicPHD_detection(input_text)

        if self.is_radar:
            self.radar_detection(input_text)

        if self.is_binoculars:
            if prompt:
                self.binoculars_detection(input_text, prompt)
            else:
                self.binoculars_detection(input_text, "")

    def get_next_token_prob(self, encoded_inputs, model_output):
        prob_list = list()
        for idx, token in enumerate(encoded_inputs["input_ids"][0]):
            probability_list = F.softmax(model_output.logits[0][idx], dim=-1)
            prob_list.append(probability_list[token].item())

        return prob_list

    def judge(self, score, threshold=0):
        judge_text = "AI-generated" if score >= threshold else "Human-generated"
        judge_label = AI_GENERATED if score >= threshold else HUMAN_GENERATED
        return judge_text, judge_label

    ## Entropy-based detection ##
    @torch.no_grad()
    def entropy_detection(self, logits):
        """
        Main processing of entropy-based detection
        """
        entropy_base_score = self.calc_entropy(logits)
        self.scores[self.entropy] = entropy_base_score

    def calc_entropy(self, logits):
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        n = len(neg_entropy[0])

        return (-(1 / n) * neg_entropy.sum(-1).sum()).item()

    def get_entropy(self, logits):
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()

    def log_p_detection(self, logits, labels):
        score = self.get_logp(logits, labels)
        self.scores[self.log_p] = score

    def max_detection(self, logits):
        score = torch.max(F.softmax(logits, dim=-1)).item()
        self.scores[self.max] = score

    def lrr_detection(self, logits, labels):
        rank = self.calc_rank(logits, labels, True)
        logp = self.get_logp(logits, labels)

        score = -np.nan_to_num(logp / rank)
        self.scores[self.lrr] = score

    def npr_detection(self, perturbed_texts, logits, labels):
        perturbed_texts_log_ranks = list()
        for t in perturbed_texts:
            encoded_inputs = self.tokenizer(t, return_tensors="pt", truncation=True).to(
                DEVICE
            )
            model_output = self.model(
                **encoded_inputs, labels=encoded_inputs["input_ids"]
            )
            perturbed_texts_log_ranks.append(
                self.calc_rank(
                    model_output.logits[:, :-1], encoded_inputs.input_ids[:, 1:]
                )
            )

        original_rank = self.calc_rank(logits, labels, True)

        score = np.nan_to_num(np.mean(perturbed_texts_log_ranks) / original_rank)

        self.scores[self.npr] = score

    def fastnpr_detection(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1

        samples = self.get_samples(logits, labels).permute([0, 2, 1])
        sample_log_ranks = list()
        for s in samples[0]:
            sample_log_ranks.append(
                self.calc_rank(logits, torch.tensor([s.tolist()], device=DEVICE), True)
            )
        log_rank = self.calc_rank(logits, labels, True)

        score = np.nan_to_num(np.mean(sample_log_ranks) / log_rank)
        self.scores[self.fastnpr] = score

    def rank_detection(self, logits, labels):
        score = self.calc_rank(logits, labels)
        self.scores[self.rank] = score

    def calc_rank(self, logits, labels, log=False):
        # retrieved from https://github.com/eric-mitchell/detect-gpt/blob/main/run.py#L298
        matches = (
            logits.argsort(-1, descending=True) == labels.unsqueeze(-1)
        ).nonzero()

        assert (
            matches.shape[1] == 3
        ), f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (
            timesteps == torch.arange(len(timesteps)).to(timesteps.device)
        ).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1  # convert to 1-indexed rank
        if log:
            ranks += torch.finfo(self.floating_point).tiny
            ranks = torch.log(ranks)

        return ranks.float().mean().item()

    def log_rank_detection(self, logits, labels):
        score = self.calc_rank(logits, labels, True)
        self.scores[self.log_rank] = score

    def roberta_detection(self, input_text):
        output = self.roberta_detector(input_text)
        score = output[0]["score"]
        if output[0]["label"] == "Real":
            score *= -1
        score = (score + 1) / 2
        self.scores[self.roberta_base] = score

    def radar_detection(self, input_text):
        encoded_input = self.radar_tokenizer(
            input_text, return_tensors="pt", truncation=True
        ).to(DEVICE)
        scores = (
            F.log_softmax(self.radar_model(**encoded_input).logits, -1)
            .exp()
            .tolist()[0]
        )
        self.scores[self.radar] = scores[0] - scores[1]

    def get_logp(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(
            -1
        )
        return log_likelihood.mean().item()

    ## Fast DetectGPT from https://github.com/baoguangsheng/fast-detect-gpt/blob/main/ ##
    def fastdetectgpt_detection(self, logits, labels):
        curvature = self.get_sampling_discrepancy(logits, logits, labels)
        self.scores[self.fastdetect_gpt] = curvature

    def get_samples(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        nsamples = self.fast_sample_num
        lprobs = torch.log_softmax(logits, dim=-1)
        distrib = torch.distributions.categorical.Categorical(logits=lprobs)
        samples = distrib.sample([nsamples])
        # if you want to use partial sampling then remove comment.
        # samples = self.replace_elements_with_probability(samples.clone(), labels, 0.8)
        samples = samples.permute([1, 2, 0])
        return samples

    def replace_elements_with_probability(self, A, B, prob):
        n = len(A)
        for i in range(n):
            num_elements = A[i][0].size(-1)
            num_replace = int(prob * num_elements)
            replace_indices = torch.randperm(num_elements)[:num_replace]
            A[i][0, replace_indices] = B[0, replace_indices]
        return A

    def get_likelihood(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
        lprobs = torch.log_softmax(logits, dim=-1)
        log_likelihood = lprobs.gather(dim=-1, index=labels)
        return log_likelihood.mean(dim=1)

    def get_sampling_discrepancy(self, logits_ref, logits_score, labels):
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        samples = self.get_samples(logits_ref, labels)
        log_likelihood_x = self.get_likelihood(logits_score, labels)
        log_likelihood_x_tilde = self.get_likelihood(logits_score, samples)
        miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
        sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
        discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
        return discrepancy.item()

    ## Detect GPT from https://github.com/eric-mitchell/detect-gpt/tree/main ##
    def detectgpt_detection(self, perturbed_texts, logits, label):
        perturbed_texts_ll = [self.get_log_likelihood(t) for t in perturbed_texts]
        original_text_ll = self.get_logp(logits, label)

        perturbation_discrepancy = original_text_ll - np.mean(perturbed_texts_ll)

        # judge_text, label = self.judge(perturbation_discrepancy)
        self.scores[self.detect_gpt] = perturbation_discrepancy

        # print(f"DetectGPT Score {perturbation_discrepancy} {judge_text}")

        # del self.mask_model
        # torch.cuda.empty_cache()

    def get_log_likelihood(self, text):
        with torch.no_grad():
            tokenized = self.tokenizer(text, return_tensors="pt", truncation=True).to(
                DEVICE
            )
            labels = tokenized.input_ids
            return -self.model(**tokenized, labels=labels).loss.item()

    def truncate_text(self, text):
        encoded_text = self.mask_tokenizer.encode(text)
        truncate_text = encoded_text[: self.max_mask_model_size]
        return self.mask_tokenizer.decode(truncate_text)

    def perturb_text(self, input_text):
        masked_text = self.tokenize_and_mask(
            input_text, self.span_size, self.pct_words_masked, False
        )
        raw_fill = self.replace_masks(masked_text)
        extracted_fill = self.extract_fills(raw_fill)
        perturbed_text = self.apply_extracted_fills(masked_text, extracted_fill)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while perturbed_text == "":
            print(f"WARNING: Texts have no fills. Trying again [attempt {attempts}].")
            masked_text = self.tokenize_and_mask(
                input_text, self.span_size, self.pct_words_masked, False
            )
            raw_fill = self.replace_masks(masked_text)
            extracted_fill = self.extract_fills(raw_fill)
            perturbed_text = self.apply_extracted_fills(masked_text, extracted_fill)
            attempts += 1

        return perturbed_text

    def tokenize_and_mask(self, text, span_length, pct, ceil_pct=False):
        tokens = text.split(" ")
        mask_string = "<<<mask>>>"

        n_spans = pct * len(tokens) / (span_length + self.buffer_size * 2)
        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        n_masks = 0
        while n_masks < n_spans:
            start = np.random.randint(0, len(tokens) - span_length)
            end = start + span_length
            search_start = max(0, start - self.buffer_size)
            search_end = min(len(tokens), end + self.buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1

        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f"<extra_id_{num_filled}>"
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = " ".join(tokens)
        return text

    def replace_masks(self, text):
        n_expected = count_masks(text)
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.mask_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        outputs = self.mask_model.generate(
            **tokens,
            max_length=150,
            do_sample=True,
            top_p=self.mask_top_p,
            num_return_sequences=1,
            eos_token_id=stop_id,
        )
        return self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]

    def extract_fills(self, text):
        # remove <pad> from beginning of each text
        text = text.replace("<pad>", "").replace("</s>", "").strip()

        # return the text in between each matched mask token
        extracted_fill = pattern.split(text)[1:-1]

        # remove whitespace around each fill
        extracted_fill = [y.strip() for y in extracted_fill]

        return extracted_fill

    def apply_extracted_fills(self, masked_text, extracted_fill):
        # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [masked_text.split(" ")]

        n_expected = count_masks(masked_text)

        # replace each mask token with the corresponding fill
        for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fill, n_expected)):
            if len(fills) < n:
                tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

        # join tokens back into text
        texts = [" ".join(x) for x in tokens]
        return texts[0]

    def intrinsicPHD_detection(self, input_text):
        score = self.get_phd_single(input_text)
        self.scores[self.intrinsicPHD] = score

    def get_phd_single(self, text):
        solver = PHD()
        # text -> preprocess_string(text)
        inputs = self.encoder_mask_tokenizer(
            preprocess_string(text),
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            outp = self.encoder_mask_model(**inputs)

        # We omit the first and last tokens (<CLS> and <SEP> because they do not directly correspond to any part of the)
        mx_points = inputs["input_ids"].shape[1] - 2

        mn_points = MIN_SUBSAMPLE
        step = (mx_points - mn_points) // INTERMEDIATE_POINTS

        new_inputs = outp[0][0].cpu().numpy()[1:-1]

        return solver.fit_transform(
            new_inputs,
            min_points=mn_points,
            max_points=mx_points - step,
            point_jump=step,
        )

    def binoculars_detection(self, input_text, prompt):
        score = self.bino.compute_score(input_text, prompt)
        self.scores[self.binoculars] = score


def count_masks(text):
    return [len([x for x in text.split() if x.startswith("<extra_id_")])]


def detect(
    file_path_human: str,
    file_path_ai: str,
    output_path: str,
    figure_output_path: str,
    model_name: str,
    is_entropy: bool = True,
    is_detectgpt: bool = True,
    is_fastdetectgpt: bool = True,
    is_lrr: bool = True,
    is_npr: bool = True,
    is_fastnpr: bool = True,
    is_roberta: bool = True,
    is_rank: bool = True,
    is_log_rank: bool = True,
    is_logp: bool = True,
    is_max: bool = True,
    is_intrinsicPHD: bool = True,
    is_radar: bool = True,
    is_binoculars: bool = False,
    datasize: int = 200,
    perturbation_num: int = 5,
    pct_words_masked: float = 0.1,
    fast_sample_num: int = 10000,
    additional_string: str = "",
    is_prompt: bool = False,
    is_estimation_prompt: bool = False,
    prompt_list: list[str] = [],
):
    try:
        human_texts, ai_texts = (
            _load_texts(file_path_human),
            _load_texts_from_json(file_path_ai),
        )
    except:
        human_texts, ai_texts = (
            _load_texts(file_path_human),
            _load_texts(file_path_ai),
        )

    ai_datasize = min(datasize, len(ai_texts))
    human_datasize = min(datasize, len(human_texts))

    ai_texts = ai_texts[:ai_datasize]
    human_texts = human_texts[:human_datasize]

    label_list = [HUMAN_GENERATED] * len(human_texts) + [AI_GENERATED] * len(ai_texts)
    all_texts = human_texts + ai_texts

    # if is_prompt:
    #     assert len(ai_texts) == len(prompt_list)

    detector = Detector(
        model_name,
        is_entropy=is_entropy,
        is_detectgpt=is_detectgpt,
        is_fastdetectgpt=is_fastdetectgpt,
        is_lrr=is_lrr,
        is_npr=is_npr,
        is_fastnpr=is_fastnpr,
        is_roberta_detector=is_roberta,
        is_rank=is_rank,
        is_log_rank=is_log_rank,
        is_log_p=is_logp,
        is_max=is_max,
        is_intrinsicPHD=is_intrinsicPHD,
        is_radar=is_radar,
        is_binoculars=is_binoculars,
        is_prompt=is_prompt,
        prompt_list=prompt_list,
        label_list=label_list,
        fast_sample_num=fast_sample_num,
        num_of_perturbation=perturbation_num,
        pct_words_masked=pct_words_masked,
    )

    output_path_model_name = model_name
    if len(model_name.split("/")) > 1:
        output_path_model_name = model_name.split("/")[-1]

    additional_output_path = f"{additional_string}_{output_path_model_name.replace('.', '_').replace('-', '_')}"
    output_path += additional_output_path
    figure_output_path += additional_output_path
    scores = dict()

    for idx, text in enumerate(tqdm(all_texts, desc="Some detector is working... :")):
        prompt_label = -1
        if is_prompt:
            if is_estimation_prompt:
                prompt_label = idx
            elif label_list[idx]:
                prompt_label = idx - len(human_texts)
        detector.detect(text, prompt_label)
        for method, score in detector.scores.items():
            scores.setdefault(method, list()).append(score)
        detector.set_score_list()

    output_json = dict()

    for method, s in scores.items():
        add_result_to_output_json(method, label_list, s, output_json)

    with open(output_path + ".json", "w") as f:
        json.dump(output_json, f, ensure_ascii=False)

    save_roc_curves(figure_output_path + ".png", output_json, model_name)


def get_prompt_all(prompt_text, text):
    return f"""[INST] <<SYS>>
    {prompt_text}
    <</SYS>>
    {text}[/INST]"""


# def get_prompt_all(prompt_text, text):
#     return prompt_text + text


def get_prompt_sum():
    prompt_text = "Would you summarize following sentences, please."

    prompt_texts = list()

    data = _load_texts("./txtdata/xsum_human_for_sum.txt")

    for value in data:
        prompt_texts.append(get_prompt_all(prompt_text, value))

    return prompt_texts


def get_prompt_estimation(file_path):
    prompt_texts = list()

    data = _load_texts("./txtdata/xsum_human_for_sum.txt")
    prompts = _load_texts_from_json(file_path)

    for p, d in zip(prompts, data):
        prompt_texts.append(get_prompt_all(p, d))

    return prompt_texts
