import torch

# gradpath = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/gradients/open_instruct/gradient/mmlu.jsonl/dim8192/all_unormalized.pt"
gradpath = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/gradients/open_instruct/gradient/gsm8k.jsonl/dim8192/grads-160.pt"
grad = torch.load(gradpath)

# gradpath2 = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/gradients/open_instruct/gradient/gsm8k.jsonl/dim8192/grads-160.pt"
# gradpath2 = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/open-instruct/datasets/megamixv2_dedup_lmsyschat_train10k.jsonl/dim8192/grads-160.pt"
# gradpath2 = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/open-instruct/datasets/megamixv2_dedup_tulu2mix-code_alpaca_train10k.jsonl/dim8192/grads-160.pt"
# gradpath2 = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/open-instruct/datasets/megamixv2_dedup_tulu2mix-flan_v2_train10k.jsonl/dim8192/grads-160.pt"
gradpath2 = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/open-instruct/datasets/megamixv2_dedup_tulu2mix-cot_train10k.jsonl/dim8192/grads-160.pt"
# gradpath2 = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/gradients/open_instruct/gradient/mmlu.jsonl/dim8192/all_unormalized.pt"
grad2 = torch.load(gradpath2)


def cos0(i, j):
    return torch.nn.functional.cosine_similarity(grad[i], grad2[j], dim=0)


def cos1(i, j):
    return torch.nn.functional.cosine_similarity(grad[i], grad[j], dim=0)


def cos2(i, j):
    return torch.nn.functional.cosine_similarity(grad2[i], grad2[j], dim=0)


def cos(i, j, g1, g2):
    return torch.nn.functional.cosine_similarity(g1[i], g2[j], dim=0)


def get_batch_grads(grad, batch_size=16):
    # sample batch_size from grad num_batches times.
    batch_grads = []
    num_batches = grad.size(0) // batch_size
    for idx in range(num_batches):
        batch_grads.append(grad[idx * batch_size: (idx + 1) * batch_size].mean(dim=0))

    return torch.stack(batch_grads)

N = 32

cos0_store, cos1_store, cos2_store = [], [], []
cos0_batch_store, cos1_batch_store, cos2_batch_store = [], [], []

grad1_batch = get_batch_grads(grad, batch_size=N)
grad2_batch = get_batch_grads(grad2, batch_size=N)
len_grad1 = grad1_batch.size(0)

for i in range(len_grad1):
    for j in range(len_grad1):
        if i != j:
            # cos0_store.append(cos0(i, j))
            # cos1_store.append(cos1(i, j))
            # cos2_store.append(cos2(i, j))
            cos0_batch_store.append(cos(i, j, grad1_batch, grad2_batch))
            cos1_batch_store.append(cos(i, j, grad1_batch, grad1_batch))
            cos2_batch_store.append(cos(i, j, grad2_batch, grad2_batch))


# quantiles.
cos0_store = torch.tensor(cos0_store)
cos1_store = torch.tensor(cos1_store)
cos2_store = torch.tensor(cos2_store)
cos0_batch_store = torch.tensor(cos0_batch_store)
cos1_batch_store = torch.tensor(cos1_batch_store)
cos2_batch_store = torch.tensor(cos2_batch_store)

# print(f"cos0: {cos0_store.quantile(0.25)}, {cos0_store.quantile(0.5)}, {cos0_store.quantile(0.75)}")
# print(f"cos1: {cos1_store.quantile(0.25)}, {cos1_store.quantile(0.5)}, {cos1_store.quantile(0.75)}")
# print(f"cos2: {cos2_store.quantile(0.25)}, {cos2_store.quantile(0.5)}, {cos2_store.quantile(0.75)}")
print(f"cross (batch): {cos0_batch_store.quantile(0.25)}, {cos0_batch_store.quantile(0.5)}, {cos0_batch_store.quantile(0.75)}")
print(f"inner 1 (batch): {cos1_batch_store.quantile(0.25)}, {cos1_batch_store.quantile(0.5)}, {cos1_batch_store.quantile(0.75)}")
print(f"inner 2 (batch): {cos2_batch_store.quantile(0.25)}, {cos2_batch_store.quantile(0.5)}, {cos2_batch_store.quantile(0.75)}")

