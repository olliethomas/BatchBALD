from typing import Optional

from blackhc.progress_bar import with_progress_bar
from torch import nn as nn
from torch.utils.data import DataLoader

import src.mc_dropout
import torch

import src.torch_utils

eval_bayesian_model_consistent_cuda_chunk_size = 1024
sampler_model_cuda_chunk_size = 1024


def eval_bayesian_model_consistent(
    bayesian_model: src.mc_dropout.BayesianModule,
    available_loader: DataLoader,
    num_classes: int,
    k: int = 20,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    global eval_bayesian_model_consistent_cuda_chunk_size

    if device is None:
        device = torch.device("cpu")

    with torch.no_grad():
        # NOTE: I'm hard-coding 10 classes here!
        B = len(available_loader.dataset)
        logits_B_K_C = torch.empty((B, k, num_classes), dtype=torch.float64)

        chunk_size = eval_bayesian_model_consistent_cuda_chunk_size if device.type == "cuda" else 64
        src.torch_utils.gc_cuda()
        k_lower = 0
        while k_lower < k:
            try:
                k_upper = min(k_lower + chunk_size, k)

                # This resets the dropout masks.
                bayesian_model.eval()

                for i, (batch, _) in enumerate(
                    with_progress_bar(available_loader, unit_scale=available_loader.batch_size)
                ):
                    lower = i * available_loader.batch_size
                    upper = min(lower + available_loader.batch_size, B)

                    batch = batch.to(device)
                    # batch_size x ws x classes
                    mc_output_B_K_C = bayesian_model(batch, k_upper - k_lower)
                    logits_B_K_C[lower:upper, k_lower:k_upper].copy_(mc_output_B_K_C.double(), non_blocking=True)  # type: ignore[attr-defined]

            except RuntimeError as exception:
                if src.torch_utils.should_reduce_batch_size(exception):
                    if chunk_size <= 1:
                        raise
                    chunk_size //= 2
                    print(f"New eval_bayesian_model_consistent_cuda_chunk_size={chunk_size} ({exception})")
                    eval_bayesian_model_consistent_cuda_chunk_size = chunk_size

                    src.torch_utils.gc_cuda()
                else:
                    raise
            else:
                k_lower += chunk_size

    return logits_B_K_C


class NoDropoutModel(nn.Module):
    def __init__(self, bayesian_net: src.mc_dropout.BayesianModule):
        super().__init__()
        self.bayesian_net = bayesian_net

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        self.bayesian_net.set_dropout_p(0)
        mc_output_B_1_C = self.bayesian_net(input, 1)
        self.bayesian_net.set_dropout_p(src.mc_dropout.DROPOUT_PROB)
        return mc_output_B_1_C.squeeze(1)


class SamplerModel(nn.Module):
    def __init__(self, bayesian_net: src.mc_dropout.BayesianModule, k: int) -> None:
        super().__init__()
        self.bayesian_net = bayesian_net
        self.num_classes = bayesian_net.num_classes
        self.k = k

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        global sampler_model_cuda_chunk_size
        if self.training:
            return src.torch_utils.logit_mean(self.bayesian_net(input, self.k), dim=1, keepdim=False)
        else:
            mc_output_B_C = torch.zeros((input.shape[0], self.num_classes), dtype=torch.float64, device=input.device)

            k = self.k

            chunk_size = sampler_model_cuda_chunk_size if input.device.type == "cuda" else 32

            k_lower = 0
            while k_lower < k:
                try:
                    k_upper = min(k_lower + chunk_size, k)

                    # Reset the mask all around.
                    self.bayesian_net.eval()

                    mc_output_B_K_C = self.bayesian_net(input, k_upper - k_lower)
                except RuntimeError as exception:
                    if src.torch_utils.should_reduce_batch_size(exception):
                        chunk_size //= 2
                        if chunk_size <= 0:
                            raise
                        if sampler_model_cuda_chunk_size != chunk_size:
                            print(f"New sampler_model_cuda_chunk_size={chunk_size} ({exception})")
                            sampler_model_cuda_chunk_size = chunk_size

                        src.torch_utils.gc_cuda()
                else:
                    mc_output_B_C += torch.sum(mc_output_B_K_C.double().exp_(), dim=1, keepdim=False)
                    k_lower += chunk_size

            return (mc_output_B_C / k).log_()
