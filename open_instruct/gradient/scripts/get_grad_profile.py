import itertools

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

train_grads = []
for dataset_name in selected_train_dataset_names:
    gradpath = f"debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/gradients/net/nfs.cirrascale/mosaic/seungjuh/open-instruct/datasets/megamixv2_dedup_{dataset_name}_train10k.jsonl/dim8192/grads-320.pt"
    grad = torch.load(gradpath)
    train_grads.append({"name": dataset_name, "grad": grad})

TEST_DATASET_DIR = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/open_instruct/gradient"
selected_test_dataset_names = [
    "gsm8k",
    "mmlu",
    "tydiqa",
    "bbh",
]
eval_grads = []
for dataset_name in selected_test_dataset_names:
    gradpath = f"debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/gradients/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/open_instruct/gradient/{dataset_name}.jsonl/dim8192/all_unormalized.pt"
    grad = torch.load(gradpath)
    eval_grads.append({"name": dataset_name, "grad": grad})


def get_batch_grads(grad, batch_size):
    # sample batch_size from grad num_batches times.
    batch_grads = []
    num_batches = grad.size(0) // batch_size
    for idx in range(num_batches):
        batch_grads.append(grad[idx * batch_size: (idx + 1) * batch_size].mean(dim=0))

    return torch.stack(batch_grads)


def cos(i, j, g1, g2):
    return torch.nn.functional.cosine_similarity(g1[i], g2[j], dim=0)


# 1. test how batched gradient similarity changes with batch size.
batch_sizes = [1, 4, 16, 64]
for train_grad in train_grads:
    print(f"Now processing {train_grad['name']}")
    for batch_size in batch_sizes:
        cos_batch_store = []
        train_grad_batch = get_batch_grads(train_grad["grad"], batch_size=batch_size)
        len_train_grad = train_grad_batch.size(0)

        for i in range(len_train_grad):
            for j in range(len_train_grad):
                if i != j:
                    cos_batch_store.append(cos(i, j, train_grad_batch, train_grad_batch))

        print(
            f"Batch size: {batch_size}, similarity: {torch.stack(cos_batch_store).quantile(0.75):.2f}")

# 2. use batch_size=64 and get the cross-dataset similarity in the train datasets.
batch_size = 64
similarity_train = {}
for train_grad1, train_grad2 in itertools.product(train_grads, train_grads):
    if train_grad1["name"] != train_grad2["name"]:
        print(f"Now processing {train_grad1['name']} and {train_grad2['name']}")
        train_grad1_batch = get_batch_grads(train_grad1["grad"], batch_size=batch_size)
        train_grad2_batch = get_batch_grads(train_grad2["grad"], batch_size=batch_size)
        len_train_grad1 = train_grad1_batch.size(0)

        cos_batch_store = []
        for i in range(len_train_grad1):
            for j in range(len_train_grad1):
                if i != j:
                    cos_batch_store.append(cos(i, j, train_grad1_batch, train_grad2_batch))

        print(
            f"Train dataset similarity: {train_grad1['name']} and {train_grad2['name']}, similarity: {torch.stack(cos_batch_store).quantile(0.75):.2f}")
        similarity_train[(train_grad1["name"], train_grad2["name"])] = torch.stack(cos_batch_store).mean()

# 2.1. use batch_size=16 and get the cross-dataset similarity in the eval datasets.
batch_size = 16
similarity_eval = {}
for train_grad1, train_grad2 in itertools.product(eval_grads, eval_grads):
    if train_grad1["name"] != train_grad2["name"]:
        print(f"Now processing {train_grad1['name']} and {train_grad2['name']}")
        train_grad1_batch = get_batch_grads(train_grad1["grad"], batch_size=batch_size)
        train_grad2_batch = get_batch_grads(train_grad2["grad"], batch_size=batch_size)
        len_train_grad1 = train_grad1_batch.size(0)

        cos_batch_store = []
        for i in range(len_train_grad1):
            for j in range(len_train_grad1):
                if i != j:
                    cos_batch_store.append(cos(i, j, train_grad1_batch, train_grad2_batch))

        print(
            f"Eval dataset similarity: {train_grad1['name']} and {train_grad2['name']}, similarity: {torch.stack(cos_batch_store).quantile(0.75):.2f}")
        similarity_train[(train_grad1["name"], train_grad2["name"])] = torch.stack(cos_batch_store).mean()

# 3. get the cross-dataset similarity in the eval datasets.
similarity_eval = {}
for eval_grad1, eval_grad2 in itertools.product(eval_grads, eval_grads):
    if eval_grad1["name"] != eval_grad2["name"]:
        print(f"Now processing {eval_grad1['name']} and {eval_grad2['name']}")
        eval_grad1_batch = get_batch_grads(eval_grad1["grad"], batch_size=1)
        eval_grad2_batch = get_batch_grads(eval_grad2["grad"], batch_size=1)
        len_eval_grad1 = eval_grad1_batch.size(0)
        len_eval_grad2 = eval_grad2_batch.size(0)

        cos_batch_store = []
        for i in range(len_eval_grad1):
            for j in range(len_eval_grad2):
                cos_batch_store.append(cos(i, j, eval_grad1_batch, eval_grad2_batch))

        print(
            f"Eval dataset similarity: {eval_grad1['name']} and {eval_grad2['name']}, similarity: {torch.stack(cos_batch_store).quantile(0.75):.2f}")
        similarity_eval[(eval_grad1["name"], eval_grad2["name"])] = torch.stack(cos_batch_store).mean()

# 4. get the cross-dataset similarity between train and eval datasets.
similarity_train_eval = {}
for train_grad, eval_grad in itertools.product(train_grads, eval_grads):
    print(f"Now processing {train_grad['name']} and {eval_grad['name']}")
    train_grad_batch = get_batch_grads(train_grad["grad"], batch_size=64)
    eval_grad_batch = get_batch_grads(eval_grad["grad"], batch_size=1)
    len_train_grad = train_grad_batch.size(0)
    len_eval_grad = eval_grad_batch.size(0)

    cos_batch_store = []
    for i in range(len_train_grad):
        for j in range(len_eval_grad):
            cos_batch_store.append(cos(i, j, train_grad_batch, eval_grad_batch))

    print(
        f"Train and eval dataset similarity: {train_grad['name']} and {eval_grad['name']}, similarity: {torch.stack(cos_batch_store).quantile(0.75):.2f}")
    similarity_train_eval[(train_grad["name"], eval_grad["name"])] = torch.stack(cos_batch_store).mean()

# 5. save the results.
torch.save(similarity_train, "similarity_train.pt")
torch.save(similarity_eval, "similarity_eval.pt")
torch.save(similarity_train_eval, "similarity_train_eval.pt")

# 6. print the results, using the pt files.
similarity_train = torch.load("similarity_train.pt")
# get the top-10 most similar pairs.
sorted_similarity_train = sorted(similarity_train.items(), key=lambda x: x[1], reverse=True)
for i in range(20):
    if i % 2 == 0:
        print(
            f"Train dataset similarity: {sorted_similarity_train[i][0]}, similarity: {sorted_similarity_train[i][1]:.2f}")

similarity_eval = torch.load("similarity_eval.pt")
# get the top-10 most similar pairs.
sorted_similarity_eval = sorted(similarity_eval.items(), key=lambda x: x[1], reverse=True)
for i in range(20):
    if i % 2 == 0:
        print(
            f"Eval dataset similarity: {sorted_similarity_eval[i][0]}, similarity: {sorted_similarity_eval[i][1]:.2f}")

similarity_train_eval = torch.load("similarity_train_eval.pt")
# get the top-10 most similar pairs.
sorted_similarity_train_eval = sorted(similarity_train_eval.items(), key=lambda x: x[1], reverse=True)
for i in range(20):
    if i % 2 == 0:
        print(
            f"Train and eval dataset similarity: {sorted_similarity_train_eval[i][0]}, similarity: {sorted_similarity_train_eval[i][1]:.2f}")
