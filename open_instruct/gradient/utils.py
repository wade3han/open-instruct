import itertools
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from trak.projectors import CudaProjector


class CombinedDataLoader:
    def __init__(self, dataloaders: list[DataLoader], mixture_weights: list[float], eval_per_steps: int,
                 smoothing_factor: float = 0.0, min_weights: float = -1.0):
        """
        Args:
            dataloaders (list): A list of DataLoader objects.
        """
        assert len(dataloaders) == len(mixture_weights)
        self.dataloaders = dataloaders
        self.mixture_weights = mixture_weights
        self.iterators = [iter(dl) for dl in dataloaders]
        self.eval_per_steps = eval_per_steps
        self.smoothing_factor = smoothing_factor
        self.min_weights = min_weights

    def __iter__(self):
        return self

    def update_mixture_weights(self, sim_matrix: torch.tensor):
        current_mixture_weights = self.mixture_weights
        print(f"Old mixture weights: {current_mixture_weights}")
        new_mixture_weights_coeff = sim_matrix.mean(dim=1).cpu().numpy()  # [num_datasets]
        current_mixture_weights = np.array(current_mixture_weights) * np.exp(
            (self.eval_per_steps / 10) * new_mixture_weights_coeff)
        current_mixture_weights = \
            (1 - self.smoothing_factor) * current_mixture_weights / current_mixture_weights.sum() + \
            self.smoothing_factor * np.ones_like(current_mixture_weights) / len(self.mixture_weights)
        current_mixture_weights /= current_mixture_weights.sum()

        if self.min_weights > -1.0:
            adjusted_weights = 0
            updated_indices = []
            for i, w in enumerate(current_mixture_weights):
                if w < self.min_weights:
                    adjusted_weights += self.min_weights - w
                    current_mixture_weights[i] = self.min_weights
                    updated_indices.append(i)

            for i in range(len(current_mixture_weights)):
                if i not in updated_indices:
                    current_mixture_weights[i] -= adjusted_weights / (
                            len(current_mixture_weights) - len(updated_indices))

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


class GradientTracker:
    def __init__(self, beta1: float, beta2: float, projector: CudaProjector, projector_batch_size: int, num_train: int):
        self.beta1 = beta1
        self.beta2 = beta2

        self.num_train = num_train

        self.gradient_store_exp_avg = {}
        self.gradient_store_exp_avg_sq = {}
        self.gradient_store_avg = {}
        self.count_per_dataset = {}
        self.batch_size = projector_batch_size
        self.projector = projector
        self.previous_full_vectorized_grads = None

        # store grads till the size becomes projector_batch_size.
        self.temporary_gradient_storage = []
        self.temporary_dataset_ids = []

    def refresh(self):
        self.temporary_gradient_storage = []
        self.temporary_dataset_ids = []
        self.previous_full_vectorized_grads = None

    def track_gradients(self, model: nn.Module, dataset_id: int, valid_num_batches: int | None = None):
        full_vectorized_grads = torch.cat(
            [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])
        if self.previous_full_vectorized_grads is None:
            residual_full_vectorized_grads = full_vectorized_grads
        else:
            residual_full_vectorized_grads = full_vectorized_grads - self.previous_full_vectorized_grads
        self.previous_full_vectorized_grads = full_vectorized_grads

        self.temporary_gradient_storage.append(residual_full_vectorized_grads)
        self.temporary_dataset_ids.append(dataset_id)

        if len(self.temporary_gradient_storage) >= self.batch_size:
            temporary_gradient_storage = torch.stack(self.temporary_gradient_storage, dim=0)
            projected_vectorized_grads = self.projector.project(
                temporary_gradient_storage.to(torch.float16).detach(),
                model_id=0,
            )

            for i, dataset_id in enumerate(self.temporary_dataset_ids):
                if self.count_per_dataset.get(dataset_id) is None:
                    self.count_per_dataset[dataset_id] = 1
                else:
                    self.count_per_dataset[dataset_id] += 1

                if valid_num_batches is not None:  # validation grads
                    if self.gradient_store_avg.get(dataset_id) is None:
                        self.gradient_store_avg[dataset_id] = projected_vectorized_grads[i]
                    else:
                        self.gradient_store_avg[dataset_id] += projected_vectorized_grads[i] / valid_num_batches

                else:
                    if self.gradient_store_exp_avg.get(dataset_id) is None:
                        self.gradient_store_exp_avg[dataset_id] = projected_vectorized_grads[i]
                    else:
                        self.gradient_store_exp_avg[dataset_id] = \
                            self.beta1 * self.gradient_store_exp_avg[dataset_id] + \
                            (1 - self.beta1) * projected_vectorized_grads[i]
                    if self.gradient_store_exp_avg_sq.get(dataset_id) is None:
                        self.gradient_store_exp_avg_sq[dataset_id] = projected_vectorized_grads[i] ** 2
                    else:
                        self.gradient_store_exp_avg_sq[dataset_id] = \
                            self.beta2 * self.gradient_store_exp_avg_sq[dataset_id] + \
                            (1 - self.beta2) * (projected_vectorized_grads[i] ** 2)

            # reset temporary storage
            self.temporary_gradient_storage = []
            self.temporary_dataset_ids = []

    def calc_sim(self, gradient_store_avg: dict[int, torch.tensor]):
        num_train_set = self.num_train
        num_valid_set = len(gradient_store_avg)

        sim_matrix = torch.zeros((num_train_set, num_valid_set))
        for train_id, val_id in itertools.product(range(num_train_set), range(num_valid_set)):
            try:
                train_grad = self.gradient_store_exp_avg[train_id] / torch.sqrt(
                    self.gradient_store_exp_avg_sq[train_id] + 1e-8)
                val_grad = gradient_store_avg[val_id]
                sim = torch.nn.functional.cosine_similarity(train_grad, val_grad, dim=0)
            except KeyError:
                sim = -1
            sim_matrix[train_id, val_id] = sim

        return sim_matrix
