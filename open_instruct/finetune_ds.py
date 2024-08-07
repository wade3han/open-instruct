# !/usr/bin/env python
# coding=utf-8
# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
import random
import time
from functools import partial

import datasets
import deepspeed
import numpy as np
import torch
import transformers
import wandb
from datasets import load_dataset
from deepspeed import get_accelerator, DeepSpeedEngine
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data._utils.fetch import _BaseDatasetFetcher
from torch.utils.data._utils.worker import _worker_loop
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Tokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    OPTForCausalLM,
    get_scheduler,
)

from open_instruct.multipack import MultipackBatchSampler, get_dataset_lengths, \
    patch_for_multipack, SUPPORTED_MULTIPACK_MODEL_TYPES, V2BatchSamplerDataCollatorForSeq2SeqPadding
from open_instruct.utils import ArgumentParserPlus, FlatArguments, MFUEstimator, print_rank_zero
from open_instruct.wsd_scheduler import get_constant_schedule_with_warmup_and_cooldown


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if isinstance(possibly_batched_index[0], list):
            data = [None for i in possibly_batched_index]
            for i, possibly_batched_index_ in enumerate(possibly_batched_index):
                if self.auto_collation:
                    if (
                            hasattr(self.dataset, "__getitems__")
                            and self.dataset.__getitems__
                    ):
                        data[i] = self.dataset.__getitems__(possibly_batched_index_)
                    else:
                        data[i] = [self.dataset[idx] for idx in possibly_batched_index_]
                else:
                    data[i] = self.dataset[possibly_batched_index_]
        else:
            if self.auto_collation:
                if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                    data = self.dataset.__getitems__(possibly_batched_index)
                else:
                    data = [self.dataset[idx] for idx in possibly_batched_index]
            else:
                data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)


def patch_fetchers():
    torch.utils.data._utils.fetch._MapDatasetFetcher = _MapDatasetFetcher
    torch.utils.data.dataloader._utils.fetch._MapDatasetFetcher = _MapDatasetFetcher


def patched_worker_loop(*args, **kwargs):
    patch_fetchers()
    return _worker_loop(*args, **kwargs)


torch.utils.data._utils.worker._worker_loop = patched_worker_loop
patch_fetchers()

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

EVAL_MAX_SEQ_LENGTH = 8192
EVAL_BATCH_SIZE = 8


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length, add_bos=False):
    """
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    """
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example["prompt"].endswith((" ", "\n", "\t")) and not example["completion"].startswith((" ", "\n", "\t")):
        example_text = example["prompt"] + " " + example["completion"]
    else:
        example_text = example["prompt"] + example["completion"]
    example_text = example_text + tokenizer.eos_token
    if add_bos:
        example_text = tokenizer.bos_token + example_text
    tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example["prompt"], return_tensors="pt", max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, : tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length, add_bos=False,
                                mask_users=True,
                                mask_padding=False, ):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    if add_bos:
        example_text = tokenizer.bos_token + example_text

    if not mask_padding:
        tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone()
    else:
        raise NotImplementedError("This is deprecated.")
        # tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True,
        #                               padding="max_length")
        # input_ids = tokenized_example.input_ids
        # labels = input_ids.clone()
        # labels[labels == tokenizer.pad_token_id] = -100

    # mask the non-assistant part for avoiding loss
    if mask_users:
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    message_start_idx = tokenizer(
                        _concat_messages(messages[:message_idx]),
                        return_tensors="pt",
                        max_length=max_seq_length,
                        truncation=True,
                    ).input_ids.shape[1]
                if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                    # here we also ignore the role of the assistant
                    messages_so_far = _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
                else:
                    messages_so_far = _concat_messages(messages[: message_idx + 1])
                message_end_idx = tokenizer(
                    messages_so_far, return_tensors="pt", max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
                labels[:, message_start_idx:message_end_idx] = -100

                if message_end_idx >= max_seq_length:
                    break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def test_model(args,
               model_engine,
               test_data_loaders: list[DataLoader],
               test_data_loaders_names: list[str],
               completed_steps: int,
               embedding_size: int,
               device: torch.device,
               ):
    model_engine.eval()
    total_eval_loss = 0
    DIVIDE_CONSTANT = EVAL_MAX_SEQ_LENGTH * EVAL_BATCH_SIZE
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for test_data_loader, dataset_name in zip(test_data_loaders, test_data_loaders_names):
            eval_loss = 0
            loss_count = 0
            for eval_batch in test_data_loader:
                eval_batch_device = {k: v.to(device) for k, v in eval_batch.items()}
                outputs = model_engine(**eval_batch_device, use_cache=False)
                loss = outputs.loss
                # logits = outputs.logits
                # labels = eval_batch["labels"]
                # # Shift so that tokens < n predict n
                # shift_logits = logits[..., :-1, :].contiguous()
                # shift_labels = labels[..., 1:].contiguous()
                # shift_logits = shift_logits.view(-1, embedding_size)
                # shift_labels = shift_labels.view(-1)
                # # Enable model parallelism
                # shift_labels = shift_labels.to(shift_logits.device)
                # loss = loss_fct(shift_logits, shift_labels)
                # loss = loss / DIVIDE_CONSTANT
                eval_loss += loss
                loss_count += 1
            torch.distributed.all_reduce(eval_loss)
            eval_loss = eval_loss / loss_count / int(os.environ["WORLD_SIZE"])
            total_eval_loss += eval_loss
            print_rank_zero(f"Eval loss for {dataset_name}: {eval_loss}")
            if args.with_tracking and int(os.environ["RANK"]) == 0:
                wandb.log({f"eval_loss_{dataset_name}": eval_loss}, step=completed_steps)
    total_eval_loss /= len(test_data_loaders)
    print_rank_zero(f"Total eval loss: {total_eval_loss}")
    if args.with_tracking and int(os.environ["RANK"]) == 0:
        wandb.log({"eval_loss": total_eval_loss}, step=completed_steps)

    model_engine.train()


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def save_model(model_engine: DeepSpeedEngine, output_dir):
    if int(os.environ["RANK"]) == 0:
        torch.save(model_engine.state_dict(), os.path.join(output_dir, "model.pth"))


def main():
    parser = ArgumentParserPlus((FlatArguments))
    args = parser.parse()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    get_accelerator().set_device(args.local_rank)
    device = torch.device(get_accelerator().device_name(), args.local_rank)
    deepspeed.init_distributed()

    if int(os.environ["RANK"]) == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if int(os.environ["RANK"]) == 0:
        print_rank_zero(f"Arguments: {args}")
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    torch.distributed.barrier()

    # hard-coded for now.
    offload = False
    zero_stage = 2

    offload_device = "cpu" if offload else "none"

    ds_config = {
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "train_batch_size": args.per_device_train_batch_size * int(os.environ["WORLD_SIZE"]),
        "zero_optimization": {
            "stage": zero_stage,
            "offload_param": {"device": offload_device},
            "offload_optimizer": {"device": offload_device},
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 3e7,
            "stage3_prefetch_bucket_size": 3e7,
            "memory_efficient_linear": False,
        },
        "bfloat16": {"enabled": True},
        "gradient_clipping": 1.0,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
            token=os.getenv("HF_TOKEN", None),
            force_download=True,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
            token=os.getenv("HF_TOKEN", None),
            force_download=True,
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    tokenizer_revision = args.model_revision if args.tokenizer_revision is None else args.tokenizer_revision

    if tokenizer_revision != args.model_revision:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tokenizer_revision}` is different
                   from the model revision `{args.model_revision}`."""
        print_rank_zero(warning)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision,
            token=os.getenv("HF_TOKEN", None),
            force_download=True,
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision,
            token=os.getenv("HF_TOKEN", None),
            force_download=True,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    with deepspeed.zero.Init(enabled=(zero_stage == 3)):
        if args.model_name_or_path:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=args.trust_remote_code,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                use_flash_attention_2=True if args.use_flash_attn else False,
                revision=args.model_revision,
                token=os.getenv("HF_TOKEN", None),
                force_download=True,
            )
        else:
            print_rank_zero("Training new model from scratch")
            model = AutoModelForCausalLM.from_config(config)

    model = torch.compile(model)

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        )
        assert num_added_tokens in [
            0,
            1,
        ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        # OLMo newer models use this tokenizer
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
            assert (
                args.add_bos
            ), "For OLMo with GPTNeoX, you must add bos token to the beginning of the input sequence."
        # else, pythia / other models
        else:
            num_added_tokens = tokenizer.add_special_tokens(
                {
                    "pad_token": "<pad>",
                }
            )
            assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({"unk_token": "<unk>"})
    elif isinstance(tokenizer, transformers.PreTrainedTokenizerFast) and tokenizer.pad_token is None:
        num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
        assert num_added_tokens == 1, "We detected no padding token but add_special_tokens did not add one."

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    # update embedding size after resizing for sum loss
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]

    # set the tokenizer chat template to the tulu format
    # this makes evaluation/etc easier down the line.
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"  # noqa: E501
    if args.add_bos:
        # also add bos in the chat template
        tokenizer.chat_template = "{{ bos_token }}" + tokenizer.chat_template

    if args.gradient_checkpointing:
        deepspeed.checkpointing.configure(mpu_=None)
        model.gradient_checkpointing = True
        model._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
                module._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
        # gradient_checkpointing_func = deepspeed.checkpointing.checkpoint
        # model._gradient_checkpointing_func = gradient_checkpointing_func
        # model.gradient_checkpointing = True
        # for module in model.modules():
        #     if hasattr(module, "gradient_checkpointing"):
        #         module.gradient_checkpointing = True
        #         module._gradient_checkpointing_func = gradient_checkpointing_func

    # Preprocessing the datasets.
    if "prompt" in raw_datasets["train"].column_names and "completion" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos=args.add_bos,
        )
    elif "messages" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            mask_users=args.mask_users,
            mask_padding=args.mask_padding,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos=args.add_bos,
        )
    else:
        raise ValueError("You need to have either 'prompt'&'completion' or 'messages' in your column names.")

    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        remove_columns=[
            name
            for name in raw_datasets["train"].column_names
            if name not in ["input_ids", "labels", "attention_mask"]
        ],
        desc="Tokenizing and reformatting instruction data",
    )

    if args.use_multipack:
        def add_position_ids(sample):
            sample_len = len(sample["input_ids"])
            sample["position_ids"] = torch.arange(len(sample["input_ids"]))
            sample["length"] = sample_len
            return sample

        lm_datasets = lm_datasets.map(
            add_position_ids,
            desc="Add position_id column (Pretraining Sample Packing)",
        )

    lm_datasets.set_format(type="pt")
    lm_datasets = lm_datasets.filter(lambda example: (example["labels"] != -100).any())

    TEST_DATASET_DIR = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct/datasets/"
    selected_validation_dataset_names = [
        "lmsyschat",
        # "tulu2mix-code_alpaca",
        # "tulu2mix-cot",
        # "tulu2mix-flan_v2",
        # "tulu2mix-gpt4_alpaca",
        # "tulu2mix-oasst1",
        # "tulu2mix-open_orca",
        # "tulu2mix-science",
        # "tulu2mix-sharegpt",
        # "tulu2mix-wizardlm",
        # "ultrachat",
        # "ultrainteract",
        # "wildchat-gpt-4-0125-preview",
    ]
    lm_datasets_tests = []
    for dataset_name in selected_validation_dataset_names:
        validation_datapath = f"{TEST_DATASET_DIR}/megamixv2_dedup_{dataset_name}_validation.jsonl"
        data_files = {"test": validation_datapath}
        raw_datasets_test = load_dataset(
            "json",
            data_files=data_files,
        )
        encode_function_mask_non_assistant = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            mask_padding=False,
            mask_users=True,
            max_seq_length=EVAL_MAX_SEQ_LENGTH,  # HARD-CODED
            add_bos=args.add_bos,
        )
        lm_datasets_test = raw_datasets_test.map(
            encode_function_mask_non_assistant,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in raw_datasets_test["test"].column_names if
                            name not in ["input_ids", "labels", "attention_mask"]],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets_test.set_format(type="pt")
        lm_datasets_tests.append(lm_datasets_test)

    train_dataset = lm_datasets["train"]
    test_datasets = [lm_datasets_test["test"] for lm_datasets_test in lm_datasets_tests]

    # debugging tool for fewer samples
    if args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), args.max_train_samples)
        print_rank_zero(f"Limiting training samples to {max_train_samples} from {len(train_dataset)}.")
        train_dataset = train_dataset.select(range(max_train_samples))

    # Log a few random samples from the training set:
    if int(os.environ["RANK"]) == 0:
        for index in random.sample(range(len(train_dataset)), 3):
            print_rank_zero(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    assert args.use_multipack, "Only multipack is supported. TODO: fix this."
    assert args.use_compile, "Multipack only works with compile. TODO: fix this."
    assert not args.mask_padding, "Mask padding is not supported with multipack."
    assert config.model_type in SUPPORTED_MULTIPACK_MODEL_TYPES, f"Model type {config.model_type} not supported."

    batch_max_len = args.per_device_train_batch_size * args.max_seq_length  # 4 * 8192 = 32768
    batch_size = 1

    sampler = MultipackBatchSampler(
        # DistributedSampler(train_dataset, num_replicas=int(os.environ["WORLD_SIZE"]), rank=int(os.environ["RANK"])),
        RandomSampler(train_dataset),
        lengths=get_dataset_lengths(train_dataset),
        packing_efficiency_estimate=1.0,
        batch_max_len=batch_max_len,
        batch_size=batch_size,
        drop_last=True,
        num_replicas=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )

    collate_fn = V2BatchSamplerDataCollatorForSeq2SeqPadding(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        max_length=batch_max_len,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
    )

    ds_config['train_micro_batch_size_per_gpu'] = batch_size  # 1
    ds_config['train_batch_size'] = batch_size * int(
        os.environ["WORLD_SIZE"]) * args.gradient_accumulation_steps  # 4 * 8 = 64

    # monkeypatch
    if args.use_flash_attn:
        patch_for_multipack(config.model_type, model_name=config._name_or_path)

    print(f"WORLD_SIZE: {int(os.environ['WORLD_SIZE'])}, RANK: {int(os.environ['RANK'])}")

    test_samplers = [MultipackBatchSampler(
        RandomSampler(test_dataset),
        lengths=get_dataset_lengths(test_dataset),
        packing_efficiency_estimate=1.0,
        batch_max_len=EVAL_MAX_SEQ_LENGTH * EVAL_BATCH_SIZE,
        batch_size=1,
        drop_last=False,
        num_replicas=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    ) for test_dataset in test_datasets]

    test_data_loaders = [
        DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            collate_fn=collate_fn,
        )
        for test_dataset, test_sampler in zip(test_datasets, test_samplers)
    ]

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    optimizer = deepspeed.ops.adam.FusedAdam(optimizer_grouped_parameters, lr=args.learning_rate,
                                             weight_decay=0.0, )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Create the learning rate scheduler.
    num_training_steps_for_scheduler = args.max_train_steps
    if args.lr_scheduler_type == "wsd":
        num_cooldown_steps = int(num_training_steps_for_scheduler * args.cooldown_ratio)
        lr_scheduler = get_constant_schedule_with_warmup_and_cooldown(
            optimizer,
            num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
            num_training_steps=num_training_steps_for_scheduler,
            num_cooldown_steps=num_cooldown_steps,
        )
    else:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_training_steps=num_training_steps_for_scheduler,
            num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
        )

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config,
    )

    # Load weights and states from a trained model, but not resuming.
    if args.load_from_checkpoint:
        print_rank_zero(f"Loading from checkpoint: {args.load_from_checkpoint}")
        model_engine.load_checkpoint(args.load_from_checkpoint)

    # # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if overrode_max_train_steps:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # # Afterwards we recalculate our number of training epochs
    # args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]
        wandb.init(project=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_ENTITY"], config=experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * int(os.environ["WORLD_SIZE"]) * \
                       args.gradient_accumulation_steps

    print_rank_zero("***** Running training *****")
    print_rank_zero(f"  Num examples = {len(train_dataset)}")
    print_rank_zero(f"  Num Epochs = {args.num_train_epochs}")
    print_rank_zero(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print_rank_zero(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print_rank_zero(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print_rank_zero(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=int(os.environ["RANK"]) != 0)
    completed_steps = 0
    starting_epoch = 0

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    mfu_estimator = MFUEstimator(config.num_hidden_layers,
                                 config.num_attention_heads,
                                 config.hidden_size,
                                 model_num_params=12 * config.num_hidden_layers * (config.hidden_size ** 2))

    t0 = time.time()
    running_emfu = -1.0
    running_mfu = -1.0
    ignore_first_few_steps_num = 4
    effective_num_tokens_per_fwdbwd = 0
    seq_length_per_fwdbwd = 0
    _loss_quantiles = None  # only available when using below_median loss masking

    for epoch in range(starting_epoch, args.num_train_epochs):
        model_engine.train()
        total_loss = 0
        active_dataloader = train_dataloader
        # # set epoch
        # train_dataloader.sampler.set_epoch(epoch)

        for step, batch in enumerate(active_dataloader):
            batch_device = {k: v.to(device) for k, v in batch.items()}
            outputs = model_engine(**batch_device, use_cache=False)
            if args.reduce_loss == "mean":
                assert args.loss_masking == "default", "mean loss only works with default loss masking"
                loss = outputs.loss
            else:
                # reduce loss is sum
                # this ensures that we weight all tokens in the dataset equally,
                # rather than weighting each overall example equally when
                # using high amounts of gradient accumulation.
                # this can result in > 5 point improvements in AlpacaEval
                # see https://github.com/huggingface/transformers/issues/24725 for
                # more discussion and details.
                logits = outputs.logits
                labels = batch_device["labels"]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_logits = shift_logits.view(-1, embedding_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
                loss = loss_fct(shift_logits, shift_labels)
                # We scale the loss based on the batch size and sequence length
                loss = loss / (args.per_device_train_batch_size * args.max_seq_length)

            model_engine.backward(loss)
            model_engine.step()

            # We keep track of the loss at each logged step
            total_loss += loss.detach().float()

            total_norm = model_engine.get_global_grad_norm()
            if hasattr(total_norm, "item"):
                total_norm = total_norm.item()

            seq_length_per_fwdbwd += batch_device["labels"].shape[-1]
            effective_num_tokens_per_fwdbwd += (batch_device["labels"] != -100).detach().sum().item()

            if model_engine.micro_steps % model_engine.gradient_accumulation_steps() == 0:
                if completed_steps % args.eval_per_steps == 0 and completed_steps > 0:
                    test_model(args, model_engine, test_data_loaders, selected_validation_dataset_names,
                               completed_steps, embedding_size, device)

                progress_bar.update(1)
                completed_steps += 1

                t1 = time.time()
                dt = t1 - t0
                t0 = t1

                if ignore_first_few_steps_num > 0:
                    emfu = -1.0
                    mfu = -1.0
                    ignore_first_few_steps_num -= 1
                else:
                    emfu = mfu_estimator.estimate_mfu(effective_num_tokens_per_fwdbwd,
                                                      dt,
                                                      int(seq_length_per_fwdbwd / args.gradient_accumulation_steps))
                    mfu = mfu_estimator.estimate_mfu(seq_length_per_fwdbwd,
                                                     # seq_length_per_fwdbwd * args.per_device_train_batch_size,
                                                     dt,
                                                     int(seq_length_per_fwdbwd / args.gradient_accumulation_steps))
                effective_num_tokens_percentage = effective_num_tokens_per_fwdbwd / \
                                                  seq_length_per_fwdbwd \
                                                  * 100
                # (seq_length_per_fwdbwd * args.per_device_train_batch_size) \
                running_emfu = emfu if running_emfu == -1.0 else 0.9 * running_emfu + 0.1 * emfu
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    torch.distributed.all_reduce(total_loss)
                    avg_loss = (
                            total_loss / int(os.environ["WORLD_SIZE"])
                            / args.gradient_accumulation_steps
                            / args.logging_steps
                    )
                    print_rank_zero(f"  Step: {completed_steps}, "
                                    f" LR: {lr_scheduler.get_last_lr()[0]}, "
                                    f" Loss: {avg_loss:.4f},"
                                    f" eMFU: {running_emfu * 100:.2f},"
                                    f" MFU: {running_mfu * 100:.2f},"
                                    f" Total Norm: {total_norm:.2f},"
                                    f" Effective Num Tokens (%): {effective_num_tokens_percentage:.2f},"
                                    f" Effective Num Tokens Per Instance: {effective_num_tokens_per_fwdbwd / args.gradient_accumulation_steps:.2f},"
                                    f" Seq Length: {seq_length_per_fwdbwd / args.gradient_accumulation_steps:.2f}")
                    if args.with_tracking and int(os.environ["RANK"]) == 0:
                        wandb.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                                "total_norm": total_norm,
                                "eMFU (%)": running_emfu * 100,
                                "MFU (%)": running_mfu * 100,
                                "effective_num_tokens (%)": effective_num_tokens_percentage,
                                "effective_num_tokens_per_instance": effective_num_tokens_per_fwdbwd / (
                                        args.per_device_train_batch_size * args.gradient_accumulation_steps),
                                "seq_length": seq_length_per_fwdbwd / args.gradient_accumulation_steps,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0

                seq_length_per_fwdbwd = 0
                effective_num_tokens_per_fwdbwd = 0

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        save_model(model_engine, output_dir)

                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            save_model(model_engine, output_dir)

    torch.distributed.barrier()
    # last evaluation
    test_model(args, model_engine, test_data_loaders, selected_validation_dataset_names,
               completed_steps, embedding_size, device)

    if args.output_dir is not None:
        if int(os.environ["RANK"]) == 0:
            tokenizer.save_pretrained(args.output_dir)
        if args.save_state:
            model_engine.save_checkpoint(args.output_dir)

    torch.distributed.barrier()


if __name__ == "__main__":
    main()
