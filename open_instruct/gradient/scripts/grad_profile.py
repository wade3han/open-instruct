import torch

gradpath = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/gradients/open_instruct/gradient/mmlu.jsonl/dim8192/all_unormalized.pt"
grad = torch.load(gradpath)

# gradpath2 = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/gradients/open_instruct/gradient/gsm8k.jsonl/dim8192/grads-160.pt"
# gradpath2 = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/open-instruct/datasets/megamixv2_dedup_lmsyschat_train10k.jsonl/dim8192/grads-160.pt"
gradpath2 = "/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/debug_results/gemma2_2b_bbh_base_lr2e-5/step_400/open-instruct/datasets/megamixv2_dedup_tulu2mix-code_alpaca_train10k.jsonl/dim8192/grads-160.pt"
grad2 = torch.load(gradpath2)


def cos0(i, j):
    return torch.nn.functional.cosine_similarity(grad[i], grad2[j], dim=0)


def cos1(i, j):
    return torch.nn.functional.cosine_similarity(grad[i], grad[j], dim=0)


def cos2(i, j):
    return torch.nn.functional.cosine_similarity(grad2[i], grad2[j], dim=0)

N = 32

cos0_store, cos1_store, cos2_store = [], [], []

for i in range(N):
    for j in range(N):
        if i != j:
            cos0_store.append(cos0(i, j))
            cos1_store.append(cos1(i, j))
            cos2_store.append(cos2(i, j))

# quantiles.
cos0_store = torch.tensor(cos0_store)
cos1_store = torch.tensor(cos1_store)
cos2_store = torch.tensor(cos2_store)

print(f"cos0: {cos0_store.quantile(0.25)}, {cos0_store.quantile(0.5)}, {cos0_store.quantile(0.75)}")
print(f"cos1: {cos1_store.quantile(0.25)}, {cos1_store.quantile(0.5)}, {cos1_store.quantile(0.75)}")
print(f"cos2: {cos2_store.quantile(0.25)}, {cos2_store.quantile(0.5)}, {cos2_store.quantile(0.75)}")

