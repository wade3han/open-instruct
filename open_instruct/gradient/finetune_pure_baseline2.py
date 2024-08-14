import itertools
import logging
import math
import os
import random
import time
from functools import partial

import datasets
import numpy as np
import torch
import transformers
import wandb
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data._utils.fetch import _BaseDatasetFetcher
from torch.utils.data._utils.worker import _worker_loop
from tqdm.auto import tqdm
from trak.projectors import CudaProjector, ProjectionType
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Tokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    get_scheduler,
    DataCollatorForSeq2Seq,
)

from open_instruct.multipack import MultipackBatchSampler, get_dataset_lengths, \
    V2BatchSamplerDataCollatorForSeq2SeqPadding, patch_for_multipack_legacy
from open_instruct.utils import ArgumentParserPlus, FlatArguments, MFUEstimator
from open_instruct.wsd_scheduler import get_constant_schedule_with_warmup_and_cooldown


def get_number_of_params(model):
    """ Make sure that only lora parameters require gradients in peft models. """
    if isinstance(model, PeftModel):
        names = [n for n, p in model.named_parameters(
        ) if p.requires_grad and "lora" not in n]
        assert len(names) == 0
    num_params = sum([p.numel()
                      for p in model.parameters() if p.requires_grad])
    print(f"Total number of parameters that require gradients: {num_params}")
    return num_params


# class CudaProjector:
#     def __init__(self, grad_dim: int, proj_dim: int, seed: int, device: torch.device,
#                  dtype: torch.dtype, block_size: int, max_batch_size: int):
#         self.grad_dim = grad_dim
#         self.proj_dim = proj_dim
#         self.seed = seed
#         self.device = device
#         self.dtype = dtype
#         self.block_size = block_size
#         self.max_batch_size = max_batch_size
#         self.proj_matrix = torch.randn((self.block_size, self.proj_dim), device=self.device, dtype=self.dtype)
#
#     @torch.no_grad()
#     def project(self, full_vectorized_grads: torch.Tensor) -> torch.Tensor:
#         """
#         Project the full vectorized gradients to the projected space.
#         """
#         # full_vectorized_grads: [batch_size, grad_dim]
#         # first, we need to pad the full_vectorized_grads to the block size.
#         # note that the grad_dim should be the multiple of the block size.
#         batch_size = full_vectorized_grads.shape[0]
#         pad_size = self.grad_dim + (self.block_size - self.grad_dim % self.block_size) % self.block_size
#         padded_full_vectorized_grads = torch.cat([
#             full_vectorized_grads,
#             torch.zeros(batch_size, pad_size - self.grad_dim, device=self.device, dtype=self.dtype)
#         ], dim=1)
#         # padded_full_vectorized_grads: [batch_size, pad_size]
#         # next, we need to reshape the padded_full_vectorized_grads to the block size.
#         reshaped_full_vectorized_grads = padded_full_vectorized_grads.view(-1, self.block_size)
#
#         # reshaped_full_vectorized_grads: [batch_size * (pad_size // block_size), block_size]
#         # finally, we need to project the reshaped_full_vectorized_grads to the projected space.
#         projected_full_vectorized_grads = torch.matmul(reshaped_full_vectorized_grads, self.proj_matrix) / math.sqrt(
#             self.proj_dim)
#         projected_full_vectorized_grads = projected_full_vectorized_grads.view(batch_size,
#                                                                                (pad_size // self.block_size),
#                                                                                self.proj_dim)
#         projected_full_vectorized_grads = projected_full_vectorized_grads.sum(dim=1)
#         # projected_full_vectorized_grads: [batch_size, proj_dim]
#         return projected_full_vectorized_grads


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
               model: nn.Module,
               test_data_loaders: list[DataLoader],
               test_data_loaders_names: list[str],
               completed_steps: int,
               embedding_size: int,
               device: torch.device,
               projector: CudaProjector,
               ):
    # model.eval()
    total_eval_loss = 0
    DIVIDE_CONSTANT = EVAL_MAX_SEQ_LENGTH * args.per_device_eval_batch_size
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")

    gradient_store_avg = {}
    count_per_dataset = {}

    for dataset_id, (test_data_loader, dataset_name) in enumerate(zip(test_data_loaders, test_data_loaders_names)):
        eval_loss = 0
        loss_count = 0
        num_batches = len(test_data_loader)

        for eval_batch in test_data_loader:
            eval_batch_device = {k: v.to(device) for k, v in eval_batch.items()}
            outputs = model(**eval_batch_device, use_cache=False)
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
            loss.backward()

            full_vectorized_grads = torch.cat(
                [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])
            projected_vectorized_grads = projector.project(
                full_vectorized_grads.to(torch.float16).unsqueeze(0).detach(),
                model_id=0,
            )
            projected_vectorized_grads = projected_vectorized_grads.squeeze(0)
            if count_per_dataset.get(dataset_id) is None:
                count_per_dataset[dataset_id] = 1
            else:
                count_per_dataset[dataset_id] += 1

            if gradient_store_avg.get(dataset_id) is None:
                gradient_store_avg[dataset_id] = projected_vectorized_grads
            else:
                gradient_store_avg[dataset_id] += projected_vectorized_grads / num_batches

            model.zero_grad()

            eval_loss += loss.detach().float()
            loss_count += 1
        eval_loss = eval_loss / loss_count
        total_eval_loss += eval_loss
        print(f"Eval loss for {dataset_name}: {eval_loss}")
        if args.with_tracking:
            wandb.log({f"eval_loss_{dataset_name}": eval_loss}, step=completed_steps)
    total_eval_loss /= len(test_data_loaders)
    print(f"Total eval loss: {total_eval_loss}")
    if args.with_tracking:
        wandb.log({"eval_loss": total_eval_loss}, step=completed_steps)

    return gradient_store_avg

    # model.train()


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


def save_model(model: nn.Module, output_dir, config, tokenizer):
    # state_dict = {}
    # for k, v in model.state_dict().items():
    #     state_dict[k.replace("module._orig_mod.", "")] = v
    #
    # model = AutoModelForCausalLM.from_config(config)
    # model.load_state_dict(state_dict)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    print(f"Arguments: {args}")
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
            token=os.getenv("HF_TOKEN", None),
            force_download=False,
            attn_implementation="flash_attention_2" if args.use_flash_attn else None,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
            token=os.getenv("HF_TOKEN", None),
            force_download=False,
            attn_implementation="flash_attention_2" if args.use_flash_attn else None,
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
        print(warning)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision,
            token=os.getenv("HF_TOKEN", None),
            force_download=False,
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision,
            token=os.getenv("HF_TOKEN", None),
            force_download=False,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        torch_dtype=torch.bfloat16,
        config=config,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        revision=args.model_revision,
        token=os.getenv("HF_TOKEN", None),
        force_download=False,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model = model.cuda()

    # FIXME: compile is not working properly.
    # if args.use_compile:
    #     model: nn.Module = torch.compile(model)

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
    elif isinstance(tokenizer, GPT2Tokenizer):  # smollm
        num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
        assert num_added_tokens == 1, "We detected no padding token but add_special_tokens did not add one."
    elif isinstance(tokenizer, transformers.PreTrainedTokenizerFast) and tokenizer.pad_token is None:
        num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
        assert num_added_tokens == 1, "We detected no padding token but add_special_tokens did not add one."

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    embedding_size = embeddings.weight.shape[0]
    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    # update embedding size after resizing for sum loss
    embeddings = model.get_input_embeddings()
    embedding_size = embeddings.weight.shape[0]

    # monkeypatch
    if args.use_flash_attn:
        patch_for_multipack_legacy(config.model_type, model_name=config._name_or_path)

    # set the tokenizer chat template to the tulu format
    # this makes evaluation/etc easier down the line.
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"  # noqa: E501
    if args.add_bos:
        # also add bos in the chat template
        tokenizer.chat_template = "{{ bos_token }}" + tokenizer.chat_template

    if args.gradient_checkpointing:
        # deepspeed.checkpointing.configure(mpu_=None)
        model.gradient_checkpointing = True
        model._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
                module._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

    # prepare training datasets.
    encode_function = partial(
        encode_with_messages_format,
        mask_users=args.mask_users,
        mask_padding=args.mask_padding,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        add_bos=args.add_bos,
    )

    def add_position_ids(sample):
        sample_len = len(sample["input_ids"])
        sample["position_ids"] = torch.arange(len(sample["input_ids"]))
        sample["length"] = sample_len
        return sample

    TRAIN_DATASET_DIR = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct/datasets/"
    selected_train_dataset_names = [
        "lmsyschat",
        "tulu2mix-code_alpaca",
        "tulu2mix-cot",
        "tulu2mix-oasst1",
        "tulu2mix-science",
    ]
    lm_datasets_trains = []
    for dataset_name in selected_train_dataset_names:
        train_datapath = f"{TRAIN_DATASET_DIR}/megamixv2_dedup_{dataset_name}_train.jsonl"
        data_files = {"train": train_datapath}
        raw_datasets_train = load_dataset(
            "json",
            data_files=data_files,
        )
        lm_datasets_train = raw_datasets_train.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[
                name
                for name in raw_datasets_train["train"].column_names
                if name not in ["input_ids", "labels", "attention_mask"]
            ],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets_train = lm_datasets_train.map(
            add_position_ids,
            desc="Add position_id column (Pretraining Sample Packing)",
        )
        lm_datasets_train.set_format(type="pt")
        lm_datasets_train = lm_datasets_train.filter(lambda example: (example["labels"] != -100).any())
        if args.max_train_samples is not None and len(lm_datasets_train["train"]) > args.max_train_samples:
            lm_datasets_train["train"] = lm_datasets_train["train"].select(range(args.max_train_samples))
        lm_datasets_trains.append(lm_datasets_train)

    TEST_DATASET_DIR = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct/datasets/"
    selected_validation_dataset_names = [
        # "lmsyschat",
        "tulu2mix-code_alpaca",
        # "tulu2mix-cot",
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

    train_datasets = [lm_datasets_train["train"] for lm_datasets_train in lm_datasets_trains]
    test_datasets = [lm_datasets_test["test"] for lm_datasets_test in lm_datasets_tests]

    # Log a few random samples from the training set:
    for train_dataset in train_datasets:
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    assert args.use_multipack, "Only multipack is supported. TODO: fix this."
    assert not args.mask_padding, "Mask padding is not supported with multipack."

    batch_max_len = args.per_device_train_batch_size * args.max_seq_length  # 4 * 8192 = 32768
    batch_size = 1

    class CombinedDataLoader:
        def __init__(self, dataloaders: list[DataLoader], mixture_weights: list[float]):
            """
            Args:
                dataloaders (list): A list of DataLoader objects.
            """
            assert len(dataloaders) == len(mixture_weights)
            self.dataloaders = dataloaders
            self.mixture_weights = mixture_weights
            self.iterators = [iter(dl) for dl in dataloaders]

        def __iter__(self):
            return self

        def update_mixture_weights(self, sim_matrix: torch.tensor):
            current_mixture_weights = self.mixture_weights
            print("Old mixture weights: {current_mixture_weights}")
            new_mixture_weights_coeff = sim_matrix.mean(dim=1).cpu().numpy()  # [num_datasets]
            current_mixture_weights = np.array(current_mixture_weights) * np.exp(new_mixture_weights_coeff)
            current_mixture_weights /= current_mixture_weights.sum()
            current_mixture_weights = current_mixture_weights.tolist()
            print(f"New mixture weights: {current_mixture_weights}")
            self.mixture_weights = current_mixture_weights

        def __next__(self) -> tuple[dict, int]:
            # Randomly select one of the dataloaders based on the mixture weights
            chosen_index = random.choices(range(len(self.dataloaders)), weights=self.mixture_weights)[0]
            chosen_iterator = self.iterators[chosen_index]

            try:
                return next(chosen_iterator), chosen_index
            except StopIteration:
                # Reset the iterator if it is exhausted
                self.iterators[chosen_index] = iter(self.dataloaders[chosen_index])
                return next(self.iterators[chosen_index]), chosen_index

        def __len__(self):
            # Define the length of the combined loader as the sum of lengths of individual dataloaders
            return sum(len(dl) for dl in self.dataloaders)

    train_data_loaders = []
    for train_dataset in train_datasets:
        sampler = MultipackBatchSampler(
            RandomSampler(train_dataset),
            lengths=get_dataset_lengths(train_dataset),
            packing_efficiency_estimate=1.0,
            batch_max_len=batch_max_len,
            batch_size=batch_size,
            drop_last=True,
            num_replicas=1,
            rank=0,
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

        train_data_loaders.append(train_dataloader)

    # def get_mixture_weights(weights: list[float]):
    #     """
    #     Normalize the weights to sum to 1.
    #     """
    #     exp_weights = [math.exp(w) for w in weights]
    #     total = sum(exp_weights)
    #     return [w / total for w in exp_weights]

    # mixture_weights = [1.0 / len(train_data_loaders) for _ in train_data_loaders]
    # mixture_weights = [0, 1, 0, 0, 0]
    mixture_weights = [0.1, 0.5, 0.1, 0.15, 0.15]
    train_dataloader = CombinedDataLoader(train_data_loaders,
                                          mixture_weights=mixture_weights)

    test_data_loaders = [
        DataLoader(
            test_dataset,
            shuffle=False,
            sampler=RandomSampler(test_dataset),
            collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
            batch_size=args.per_device_eval_batch_size,
        )
        for test_dataset in test_datasets
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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

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
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_datasets)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    forward_steps = 0
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

    gradient_store_exp_avg, gradient_store_exp_avg_sq = {}, {}
    count_per_dataset = {}
    number_of_params = get_number_of_params(model)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    block_size = 128
    proj_dim = 512
    projector_batch_size = 16

    # projector = BasicProjector(grad_dim=number_of_params,
    #                            proj_dim=8192,
    #                            seed=args.seed,
    #                            proj_type=ProjectionType.rademacher,
    #                            device=torch.device('cpu'),
    #                            dtype=torch.float32,
    #                            block_size=block_size,
    #                            )
    projector = CudaProjector(grad_dim=number_of_params,
                              proj_dim=proj_dim,
                              seed=args.seed,
                              device=device,
                              dtype=torch.float16,
                              block_size=block_size,
                              proj_type=ProjectionType.rademacher,
                              max_batch_size=projector_batch_size)
    previous_projected_vectorized_grads = None

    assert args.per_device_train_batch_size == 1, "Only per_device_train_batch_size == 1 is supported."

    def calc_sim(gradient_store_avg, gradient_store_exp_avg, gradient_store_exp_avg_sq):
        num_train_set = len(gradient_store_exp_avg)
        num_valid_set = len(gradient_store_avg)

        sim_matrix = torch.zeros((num_train_set, num_valid_set))
        for train_id, val_id in itertools.product(range(num_train_set), range(num_valid_set)):
            train_grad = gradient_store_exp_avg[train_id] / torch.sqrt(gradient_store_exp_avg_sq[train_id] + 1e-8)
            val_grad = gradient_store_avg[val_id]
            sim = torch.nn.functional.cosine_similarity(train_grad, val_grad, dim=0)
            sim_matrix[train_id, val_id] = sim

        return sim_matrix

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0

        for step, (batch, dataset_id) in enumerate(train_dataloader):
            batch_device = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch_device, use_cache=False)
            forward_steps += 1
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

            (loss / args.gradient_accumulation_steps).backward()

            full_vectorized_grads = torch.cat(
                [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])
            projected_vectorized_grads = projector.project(
                full_vectorized_grads.to(torch.float16).unsqueeze(0).detach(),
                model_id=0,
            )
            projected_vectorized_grads = projected_vectorized_grads.squeeze(0)
            if count_per_dataset.get(dataset_id) is None:
                count_per_dataset[dataset_id] = 1
            else:
                count_per_dataset[dataset_id] += 1

            if previous_projected_vectorized_grads is not None:
                residual_projected_vectorized_grads = projected_vectorized_grads - previous_projected_vectorized_grads  # on cuda.
            else:
                residual_projected_vectorized_grads = projected_vectorized_grads

            if gradient_store_exp_avg.get(dataset_id) is None:
                gradient_store_exp_avg[dataset_id] = residual_projected_vectorized_grads
            else:
                gradient_store_exp_avg[dataset_id] = args.beta1 * gradient_store_exp_avg[dataset_id] + (
                        1 - args.beta1) * residual_projected_vectorized_grads

            if gradient_store_exp_avg_sq.get(dataset_id) is None:
                gradient_store_exp_avg_sq[dataset_id] = residual_projected_vectorized_grads ** 2
            else:
                gradient_store_exp_avg_sq[dataset_id] = args.beta2 * gradient_store_exp_avg_sq[dataset_id] + (
                        1 - args.beta2) * residual_projected_vectorized_grads ** 2

            previous_projected_vectorized_grads = projected_vectorized_grads

            if forward_steps % args.gradient_accumulation_steps == 0:
                # get clip_grad_norm
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                if hasattr(total_norm, "item"):
                    total_norm = total_norm.item()

                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()

            # We keep track of the loss at each logged step
            total_loss += loss.detach().float()

            seq_length_per_fwdbwd += batch_device["labels"].shape[-1]
            effective_num_tokens_per_fwdbwd += (batch_device["labels"] != -100).detach().sum().item()

            if forward_steps % args.gradient_accumulation_steps == 0:  # accumulation
                if completed_steps % args.eval_per_steps == 0 and completed_steps > 0:
                    gradient_store_avg = test_model(args, model, test_data_loaders, selected_validation_dataset_names,
                                                    completed_steps, embedding_size, device, projector)

                    # calculate the similarity.
                    sim_matrix_2by2 = calc_sim(gradient_store_avg, gradient_store_exp_avg, gradient_store_exp_avg_sq)
                    print("Similarity Matrix in the training step: ", completed_steps)

                    # use sim matrix to update the data weights.
                    # if args.reweighting:
                    #     train_dataloader.update_mixture_weights(sim_matrix_2by2)
                    #     mixture_weights = train_dataloader.mixture_weights
                    #     print(f"Updated mixture weights: {mixture_weights}")

                    if args.with_tracking:
                        wandb.log({f"mixture_weights_{i}": w for i, w in enumerate(mixture_weights)},
                                  step=completed_steps)

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
                    avg_loss = (
                            total_loss
                            / args.gradient_accumulation_steps
                            / args.logging_steps
                    )
                    print(f"  Step: {completed_steps}, "
                          f" LR: {lr_scheduler.get_last_lr()[0]}, "
                          f" Loss: {avg_loss:.4f},"
                          f" eMFU: {running_emfu * 100:.2f},"
                          f" MFU: {running_mfu * 100:.2f},"
                          f" Total Norm: {total_norm:.2f},"
                          f" Effective Num Tokens (%): {effective_num_tokens_percentage:.2f},"
                          f" Effective Num Tokens Per Instance: {effective_num_tokens_per_fwdbwd / args.gradient_accumulation_steps:.2f},"
                          f" Seq Length: {seq_length_per_fwdbwd / args.gradient_accumulation_steps:.2f}")
                    if args.with_tracking:
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
                        save_model(model, output_dir, model.config, tokenizer)

                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            save_model(model, output_dir, model.config, tokenizer)

    # last evaluation
    test_model(args, model, test_data_loaders, selected_validation_dataset_names,
               completed_steps, embedding_size, device, projector)

    if args.output_dir is not None:
        save_model(model, args.output_dir, model.config, tokenizer)
        if args.save_state:
            model.save_checkpoint(args.output_dir)


if __name__ == "__main__":
    main()
