from torch.utils.data import Subset, Dataset
import numpy as np


# TODO: I fucked this one up. Get rid of this again. (Need the range here to support slicing!)
from typing import List


def SubrangeDataset(dataset: Dataset, begin: int, end: int) -> Subset:
    if end > len(dataset):
        end = len(dataset)
    return Subset(dataset, range(begin, end))


def dataset_subset_split(dataset: Dataset, indices: List[int]) -> List[Subset]:
    if isinstance(indices, int):  # type: ignore[unreachable]
        indices = [indices]  # type: ignore[unreachable]

    datasets = []

    last_index = 0
    for index in indices:
        datasets.append(SubrangeDataset(dataset, last_index, index))
        last_index = index
    datasets.append(SubrangeDataset(dataset, last_index, len(dataset)))

    return datasets
