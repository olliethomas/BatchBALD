import enum
from typing import Optional

import torch
from torch import nn as nn

import src.independent_batch_acquisition
import src.multi_bald
from .acquisition_batch import AcquisitionBatch
from .acquisition_functions import AcquisitionFunction


class AcquisitionMethod(enum.Enum):
    independent = "independent"
    multibald = "multibald"

    def acquire_batch(
        self,
        bayesian_model: src.mc_dropout.BayesianModule,
        acquisition_function: AcquisitionFunction,
        available_loader: torch.utils.data.DataLoader,
        num_classes: int,
        k: int,
        b: int,
        min_candidates_per_acquired_item: int,
        min_remaining_percentage: float,
        initial_percentage: int,
        reduce_percentage: int,
        device: Optional[torch.device] = None,
    ) -> AcquisitionBatch:
        target_size = max(
            min_candidates_per_acquired_item * b, len(available_loader.dataset) * min_remaining_percentage // 100
        )

        if self == self.independent:  # type: ignore[comparison-overlap]
            return src.independent_batch_acquisition.compute_acquisition_bag(
                bayesian_model=bayesian_model,
                acquisition_function=acquisition_function,
                num_classes=num_classes,
                k=k,
                b=b,
                initial_percentage=initial_percentage,
                reduce_percentage=reduce_percentage,
                available_loader=available_loader,
                device=device,
            )
        # This seems to be the default used in experiment
        elif self == self.multibald:  # type: ignore[comparison-overlap]
            return src.multi_bald.compute_multi_bald_batch(
                bayesian_model=bayesian_model,
                available_loader=available_loader,
                num_classes=num_classes,
                k=k,
                b=b,
                initial_percentage=initial_percentage,
                reduce_percentage=reduce_percentage,
                target_size=target_size,  # type: ignore[arg-type]
                device=device,
            )
        else:
            raise NotImplementedError(f"Unknown acquisition method {self}!")
