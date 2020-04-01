from typing import Optional, Any, Tuple

from torch import Tensor
from torch.utils import data as data
from torch.utils.data import Dataset


class TransformedDataset(data.Dataset):
    """
    Transforms a dataset.

    Arguments:
        dataset (Dataset): The whole Dataset
        transformer (LambdaType): (idx, sample) -> transformed_sample
    """

    def __init__(
        self, dataset: Dataset, *, transformer: Optional[Any] = None, vision_transformer: Optional[Any] = None
    ) -> None:
        self.dataset = dataset
        assert not transformer or not vision_transformer
        if transformer:
            self.transformer = transformer
        else:
            assert vision_transformer is not None
            self.transformer = lambda _, data_label: (vision_transformer(data_label[0]), data_label[1])

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        return self.transformer(idx, self.dataset[idx])

    def __len__(self) -> int:
        return len(self.dataset)
