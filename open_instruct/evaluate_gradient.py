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
import os
from collections import defaultdict
from datetime import timedelta
from functools import partial

import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    GPT2Tokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    OPTForCausalLM,
)

from open_instruct.utils import ArgumentParserPlus, FlatArguments

logger = get_logger(__name__)

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

EVAL_MAX_SEQ_LENGTH = 2048
EVAL_BATCH_SIZE = 1


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


def measure_gradient(args,
                     model,
                     test_data_loaders: list[DataLoader],
                     test_data_loaders_names: list[str],
                     accelerator,
                     # optimizer,
                     ):
    total_eval_loss = 0
    DIVIDE_CONSTANT = EVAL_MAX_SEQ_LENGTH * EVAL_BATCH_SIZE
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")

    for test_data_loader, dataset_name in zip(test_data_loaders, test_data_loaders_names):
        print(f"***** Running Evaluation for {dataset_name} *****")
        loss_count = 0
        grad_per_params = defaultdict(list)
        for eval_batch in test_data_loader:
            outputs = model(**eval_batch, use_cache=False)
            loss = outputs.loss
            loss_count += 1
            loss.backward()
            # accelerator.backward(loss)
            # accelerator.wait_for_everyone()

            # get the gradient norm for the all parameters
            # this is a list of tensors, one for each parameter group
            if accelerator.sync_gradients:
                for n, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    grad = p.grad
                    print(grad)
                    print(grad.detach().cpu().numpy().flatten())
                    # flatten the gradient tensor
                    grad_per_params[n].append(grad.detach().cpu().numpy().flatten())
            # zero the gradients
            model.zero_grad()
            # optimizer.zero_grad()

            if loss_count == 3:
                break

        # get the average gradient norm for each parameter group
        acc_grad_per_params = {}
        for n in grad_per_params:
            acc_grad_per_params[n] = np.mean(np.stack(grad_per_params[n], axis=0))

        # check whether different devices have the same gradient norm
        for i, n in enumerate(acc_grad_per_params):
            print(f"Gradient for {n}: {acc_grad_per_params[n]} in RANK {accelerator.local_process_index}")
            if i == 3:
                break


def main():
    parser = ArgumentParserPlus((FlatArguments))
    args = parser.parse()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"Arguments: {args}")

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
        logger.warn(warning)

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

    if args.model_name_or_path:
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            device_map = {"": device_index}  # force data-parallel training.
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                load_in_4bit=True,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True if args.use_flash_attn else False,
                revision=args.model_revision,
                token=os.getenv("HF_TOKEN", None),
                force_download=True,
            )
        else:
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
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

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

    # We resize the embeddings only when necessary to avoid index errors.
    embeddings = model.get_input_embeddings()
    embedding_size = embeddings.weight.shape[0]
    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    embeddings = model.get_input_embeddings()
    embedding_size = embeddings.weight.shape[0]

    # set the tokenizer chat template to the tulu format
    # this makes evaluation/etc easier down the line.
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"  # noqa: E501
    if args.add_bos:
        # also add bos in the chat template
        tokenizer.chat_template = "{{ bos_token }}" + tokenizer.chat_template

    with accelerator.main_process_first():
        # TEST_DATASET_PATH = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct/datasets/tulu-v2-sft-mixture-held-out.jsonl"
        TEST_DATASET_DIR = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct/datasets/"
        selected_validation_dataset_names = [
            "lmsyschat",
            "tulu2mix-code_alpaca",
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
            lm_datasets_test = lm_datasets_test.filter(lambda example: (example["labels"] != -100).any())
            lm_datasets_tests.append(lm_datasets_test)

    test_datasets = [lm_datasets_test["test"] for lm_datasets_test in lm_datasets_tests]

    test_data_loaders = [
        DataLoader(
            test_dataset,
            shuffle=False,
            collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
            batch_size=EVAL_BATCH_SIZE,
        )
        for test_dataset in test_datasets
    ]

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]
        accelerator.init_trackers(
            os.environ["WANDB_PROJECT"], experiment_config,
            init_kwargs={"wandb": {"entity": os.environ["WANDB_ENTITY"]}}
        )
    #
    # # Optimizer
    # # Split weights in two groups, one with weight decay and the other not.
    # no_decay = ["bias", "layer_norm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    # ]
    # if args.use_qlora:
    #     from bitsandbytes.optim import AdamW
    #
    #     optimizer = AdamW(
    #         optimizer_grouped_parameters,
    #         lr=args.learning_rate,
    #         optim_bits=8 if args.use_8bit_optimizer else 32,
    #         is_paged=True,
    #     )
    # else:
    #     optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with `accelerator`.
    model, *test_data_loaders = accelerator.prepare(
        model, *test_data_loaders,
    )

    # last evaluation
    accelerator.print("***** Running Evaluation *****")
    measure_gradient(args, model, test_data_loaders, selected_validation_dataset_names, accelerator,
                     )
    # optimizer, )
    accelerator.print("***** Evaluation finished *****")


if __name__ == "__main__":
    main()
