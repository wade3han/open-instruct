import torch

TRAIN_DATASET_DIR = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct/datasets/"
selected_train_dataset_names = [
    "lmsyschat",
    "tulu2mix-code_alpaca",
    "tulu2mix-cot",
    "tulu2mix-flan_v2",
    "tulu2mix-gpt4_alpaca",
    "tulu2mix-oasst1",
    "tulu2mix-open_orca",
    "tulu2mix-science",
    "tulu2mix-sharegpt",
    "tulu2mix-wizardlm",
    "ultrachat",
    "ultrainteract",
    "wildchat-gpt-4-0125-preview",
]

grads = []
for dataset_name in selected_train_dataset_names:
    gradpath = f"/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/open-instruct/datasets/megamixv2_dedup_{dataset_name}_train10k.jsonl/dim8192/grads-320.pt"
    grad = torch.load(gradpath)
    grads.append({"name": dataset_name, "grad": grad})

TEST_DATASET_DIR = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/open_instruct/gradient"
selected_test_dataset_names = [
    "gsm8k",
    "mmlu",
    "tydiqa",
    "bbh",
]
for dataset_name in selected_test_dataset_names:
    gradpath = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/gradients/open_instruct/gradient/gsm8k.jsonl/dim8192/grads-160.pt"
    test_datapath = f"{TEST_DATASET_DIR}/{dataset_name}.jsonl"

# 1. test how batched gradient similarity changes with batch size.
batch_sizes = [1, 4, 16, 64, 256]

