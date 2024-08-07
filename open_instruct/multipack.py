"""
Multipack Batch Sampler, implemented in https://github.com/axolotl-ai-cloud/axolotl
multipack patching for v2 of sample packing
"""
import importlib
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Union
from typing import Iterable, List

import numba
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import init_empty_weights
from datasets import Dataset
# from axolotl.monkeypatch.mixtral import patch_mixtral_moe_forward_zero3
from torch.utils.data import BatchSampler, Sampler
from transformers import AutoConfig, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

# from transformers import DataCollatorForSeq2Seq

LOG = logging.getLogger(__name__)


def get_dataset_lengths(dataset: Dataset) -> np.ndarray:
    input_ids = dataset["input_ids"]
    lengths = [len(x) for x in input_ids]
    lengths = np.array(lengths, dtype=np.int64)
    # this caused a bug in the original code
    # input_ids = dataset.data.column("input_ids")
    # lengths = np.vectorize(len)(np.array(input_ids, dtype=object))
    # return lengths
    return lengths


@numba.njit
def ffd_check(a: np.ndarray, c: int, n: int) -> bool:
    # First-fit-decreasing bin packing
    # Check if a[] could fit in n bins with capacity c
    # https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing

    a = np.sort(a)[::-1]
    bins = np.full((n,), c, dtype=a.dtype)
    for size in a:
        not_found = True
        for idx in range(n):
            if bins[idx] >= size:
                bins[idx] -= size
                not_found = False
                break

        if not_found:
            return False

    return True


@numba.njit
def ffd_with_result(a: np.ndarray, c: int, start_index: int):
    # First-fit-decreasing bin packing (with result return)

    indices = np.argsort(a)[::-1]
    a = a[indices]

    bins: List[Any] = []
    bins_result: List[Any] = []
    for a_id, size in enumerate(a):
        add_new = True
        for idx in range(len(bins)):
            if bins[idx] >= size:
                bins[idx] -= size
                bins_result[idx].append(indices[a_id] + start_index)
                add_new = False
                break

        if add_new:
            bins.append(c - size)
            bins_result.append([indices[a_id] + start_index])

    return bins_result


@numba.njit
def allocate(
        lengths: np.ndarray, lengths_cumsum: np.ndarray, rank: int, c: int, n: int
):
    # Dynamic batch allocator, similar to Multifit
    # https://en.wikipedia.org/wiki/Multifit_algorithm
    # ~99.5% efficiency on OpenChat training set (12 * 2048 ctx len)

    s = 0
    start_index = 0
    result = []

    while True:
        # binary search [l, r)
        left = 1
        right = 1 + np.searchsorted(lengths_cumsum[start_index:], s + c * n, "right")

        while right - left > 1:
            mid = (left + right) // 2
            if ffd_check(lengths[start_index: start_index + mid], c, n):
                left = mid
            else:
                right = mid

        # use length l
        batch = ffd_with_result(
            lengths[start_index: start_index + left], c, start_index
        )
        assert len(batch) <= n
        if len(batch) < n:
            break

        start_index += left
        s = lengths_cumsum[start_index - 1]

        # add local rank
        result.append(batch[rank])

    return result, s, len(result) * c * n


class MultipackBatchSampler(BatchSampler):
    """
    Batch Sampler class for multipack
    """

    def __init__(
            self,
            sampler: Union[Sampler[int], Iterable[int]],
            batch_size: int,
            batch_max_len: int,
            lengths: np.ndarray,
            packing_efficiency_estimate: float = 1.0,
            drop_last: bool = False,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
    ):
        super().__init__(sampler, batch_size, drop_last)
        self.batch_size = batch_size
        self.batch_max_len = batch_max_len
        self.lengths: np.ndarray = lengths
        self.packing_efficiency_estimate = packing_efficiency_estimate or 1.0

        assert isinstance(self.lengths, np.ndarray)

        self.epoch = 0

        # statistics
        self.eff_total_used = 0
        self.eff_total_slots = 0

        # distributed
        self.num_replicas = num_replicas
        self.rank = rank

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def generate_batches(self, set_stats=False) -> list[list[list[int]]]:
        indices = [idx for idx in self.sampler]

        lengths = self.lengths[indices]
        lengths_cumsum = np.cumsum(lengths)

        batches, total_used, total_slots = allocate(
            lengths=lengths,
            lengths_cumsum=lengths_cumsum,
            rank=0,
            c=self.batch_max_len,
            n=1,
        )

        generated_batches = [
            [
                [indices[b_idx] for b_idx in batch]
                for batch in batches[i: i + self.batch_size]
            ]
            for i in range(0, len(batches), self.batch_size)
        ]

        if len(generated_batches[-1]) < self.batch_size:
            generated_batches.pop()
        if self.num_replicas is not None and len(generated_batches) % self.num_replicas != 0:
            # pop the last batch if it is not divisible by num_replicas
            overflow = len(generated_batches) % self.num_replicas
            for _ in range(overflow):
                generated_batches.pop()

        # statistics
        if set_stats:
            self.eff_total_used += total_used
            self.eff_total_slots += total_slots

        if self.num_replicas is None:
            return generated_batches
        else:  # distributed
            return generated_batches[self.rank::self.num_replicas]

    def __iter__(self):
        batches = self.generate_batches(set_stats=True)
        return iter(batches)

    def num_batches(self):
        batches = self.generate_batches(set_stats=True)
        return len(batches)

    def efficiency(self):
        return self.eff_total_used / self.eff_total_slots

    def __len__(self):
        self.num_batches()
        return self._len_est()

    def _len_est(self):
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        lengths_sum = np.sum(self.lengths)
        lengths_sum_per_device = lengths_sum // world_size
        LOG.info(
            f"packing_efficiency_estimate: {self.packing_efficiency_estimate} "
            f"total_num_tokens per device: {lengths_sum_per_device}"
        )

        # shave off 1% + 1 for dealing with variance in packing from random sampler to sampler
        return max(
            0,
            (
                    world_size
                    * math.floor(
                0.99
                * lengths_sum_per_device
                / self.packing_efficiency_estimate
                // (self.batch_max_len * self.batch_size)
            )
                    - 1
            ),
        )


@dataclass
class V2BatchSamplerDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """
    Collator for multipack specific to the using the BatchSampler
    """

    def __call__(self, features, return_tensors=None):
        if not isinstance(features[0], list):
            features = [features]
        out_features = [{} for _ in features]
        for i, features_ in enumerate(features):
            for feature in features_[0].keys():
                if feature == "length":
                    continue
                if feature == "attention_mask":
                    arrays = [
                        (i + 1) * np.array(item[feature])
                        for i, item in enumerate(features_)
                        if feature in item
                    ]
                    out_features[i][feature] = np.concatenate(arrays)
                else:
                    arrays = [
                        np.array(item[feature]) for item in features_ if feature in item
                    ]
                    out_features[i][feature] = np.concatenate(arrays)
        return super().__call__(out_features, return_tensors=return_tensors)


@dataclass
class V2BatchSamplerDataCollatorForSeq2SeqPadding(DataCollatorForSeq2Seq):
    """
    Collator for multipack specific to the using the BatchSampler
    """

    def __call__(self, features, return_tensors=None):
        if not isinstance(features[0], list):
            features = [features]
        out_features = [{} for _ in features]
        for i, features_ in enumerate(features):
            for feature in features_[0].keys():
                if feature == "length":
                    continue
                if feature == "attention_mask":
                    arrays = [
                        (i + 1) * np.array(item[feature])
                        for i, item in enumerate(features_)
                        if feature in item
                    ]
                    concat_arrays = np.concatenate(arrays)
                    # if shorter than max length, pad
                    if len(concat_arrays) < self.max_length:
                        pad_length = self.max_length - len(concat_arrays)
                        concat_arrays = np.concatenate(
                            [concat_arrays, np.zeros(pad_length, dtype=np.int64)]
                        )
                    out_features[i][feature] = concat_arrays
                elif feature in ["input_ids", "position_ids"]:
                    arrays = [
                        np.array(item[feature]) for item in features_ if feature in item
                    ]
                    concat_arrays = np.concatenate(arrays)
                    # if shorter than max length, pad
                    if len(concat_arrays) < self.max_length:
                        pad_length = self.max_length - len(concat_arrays)
                        concat_arrays = np.concatenate(
                            [concat_arrays, np.zeros(pad_length, dtype=np.int64)]
                        )
                    out_features[i][feature] = concat_arrays
                elif feature == "labels":
                    arrays = [
                        np.array(item[feature]) for item in features_ if feature in item
                    ]
                    concat_arrays = np.concatenate(arrays)
                    # if shorter than max length, pad
                    if len(concat_arrays) < self.max_length:
                        pad_length = self.max_length - len(concat_arrays)
                        concat_arrays = np.concatenate(
                            [concat_arrays, np.ones(pad_length, dtype=np.int64) * -100]
                        )
                    out_features[i][feature] = concat_arrays
                else:
                    raise ValueError(f"Unsupported feature: {feature}")
        return super().__call__(out_features, return_tensors=return_tensors)


SUPPORTED_MULTIPACK_MODEL_TYPES = [
    "llama",
    "mixtral",
    "qwen2",
    "qwen2_moe",
    "falcon",
    "phi",
    "gemma",
    "gemma2",
    "gemmoe",
    "starcoder2",
    "deepseek_v2",
]


@torch.jit.script
def get_max_seqlen_in_batch(attention_mask: torch.Tensor) -> torch.Tensor:
    max_num = int(torch.max(attention_mask).item())
    batch_size, _ = attention_mask.shape
    counts = torch.zeros((batch_size, max_num), dtype=torch.int32)

    for i in range(1, max_num + 1):
        mask = attention_mask == i
        counts[:, i - 1] = torch.sum(mask, dim=-1).to(dtype=torch.int32)

    result = counts.flatten()
    nonzero_indices = torch.nonzero(result).squeeze(-1)
    return result[nonzero_indices]


@torch.jit.script
def get_unpad_data(attention_mask: torch.Tensor):
    # https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_flash_attention_utils.py
    # fix for multipack
    device = attention_mask.device
    seqlens_in_batch = get_max_seqlen_in_batch(attention_mask)
    indices = torch.nonzero(attention_mask.flatten()).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = (
        F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        .to(device=device)
        .detach()
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def patch_for_multipack(model_type, model_name=None):
    # if model_type == "mixtral":
    #     transformers.models.mixtral.modeling_mixtral._get_unpad_data = (  # pylint: disable=protected-access
    #         get_unpad_data
    #     )
    #     if is_deepspeed_zero3_enabled():
    #         patch_mixtral_moe_forward_zero3()
    if model_type == "llama":
        transformers.models.llama.modeling_llama._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
    elif model_type == "qwen2":
        transformers.models.qwen2.modeling_qwen2._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
    elif model_type == "qwen2_moe":
        transformers.models.qwen2_moe.modeling_qwen2_moe._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
    elif model_type == "falcon":
        transformers.models.falcon.modeling_falcon._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
    elif model_type == "phi":
        transformers.models.phi.modeling_phi._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
    elif model_type == "gemma":
        transformers.models.gemma.modeling_gemma._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
    elif model_type == "gemma2":
        transformers.models.gemma2.modeling_gemma2._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
    elif model_type == "starcoder2":
        transformers.models.starcoder2.modeling_starcoder2._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
    elif model_type == "gemmoe":
        patch_remote(model_name, ".configuration_gemmoe", ".modeling_gemmoe")
    elif model_type == "jamba":
        patch_remote(model_name, ".configuration_jamba", ".modeling_jamba")
    elif model_type == "deepseek_v2":
        patch_remote(model_name, ".configuration_deepseek", ".modeling_deepseek")


def patch_remote(model_name, config_name, modeling_name):
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # we need to load the model here in order for modeling_* to be available
    with init_empty_weights():
        AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    module_name = model_config.__class__.__module__.replace(config_name, modeling_name)
    modeling_arch = importlib.import_module(module_name)
    modeling_arch._get_unpad_data = get_unpad_data  # pylint: disable=protected-access
