import os

script_format = "python open_instruct/gradient/baselines/less.py eval \
--dataset_name {dataset_name} \
--max_num_samples 320 \
--model_path debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/"

TRAIN_DATASET_DIR = "debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/gradients/net/nfs.cirrascale/mosaic/seungjuh/open-instruct/datasets"
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
lm_datasets_trains = []
for dataset_name in selected_train_dataset_names:
    train_datapath = f"{TRAIN_DATASET_DIR}/megamixv2_dedup_{dataset_name}_train10k.jsonl"
    script = script_format.format(dataset_name=train_datapath)
    os.system(script)
    print(f"Finished {dataset_name}")

TEST_DATASET_DIR = "debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/gradients/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/open_instruct/gradient/"
selected_test_dataset_names = [
    "gsm8k",
    "mmlu",
    "tydiqa",
    "bbh",
]
lm_datasets_tests = []
for dataset_name in selected_test_dataset_names:
    test_datapath = f"{TEST_DATASET_DIR}/{dataset_name}.jsonl"
    script = script_format.format(dataset_name=test_datapath)
    os.system(script)
    print(f"Finished {dataset_name}")
