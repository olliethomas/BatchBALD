import collections
import enum
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Counter, Dict, Iterator, List, Optional, Tuple

import pandas
import sklearn
import torch
import torch.utils.data as data
from ethicml.data import Adult, load_data
from ethicml.preprocessing import train_test_split
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import Compose

import src.models.adult_model
import src.models.emnist_model
import src.models.mnist_model
import src.models.vgg_model
import src.subrange_dataset

from .active_learning_data import ActiveLearningData
from .torch_utils import get_balanced_sample_indices
from .train_model import train_model
from .transformed_dataset import TransformedDataset


@dataclass
class ExperimentData:
    active_learning_data: ActiveLearningData
    train_dataset: Dataset
    available_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset
    initial_samples: List[int]


@dataclass
class DataSource:
    train_dataset: Dataset
    validation_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    shared_transform: Optional[Any] = None
    train_transform: Optional[Any] = None
    scoring_transform: Optional[Any] = None


def get_CINIC10(root: str = "./") -> DataSource:
    cinic_directory = root + "data/CINIC-10"
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    shared_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cinic_mean, std=cinic_std)])

    train_dataset = datasets.ImageFolder(cinic_directory + "/train")
    validation_dataset = datasets.ImageFolder(cinic_directory + "/valid")

    # Concatenate train and validation set to have more samples.
    merged_train_dataset = torch.utils.data.ConcatDataset([train_dataset, validation_dataset])

    test_dataset = datasets.ImageFolder(cinic_directory + "/test")

    return DataSource(
        train_dataset=merged_train_dataset,
        test_dataset=test_dataset,
        shared_transform=shared_transform,
        train_transform=train_transform,
    )


def get_Adult() -> DataSource:
    data = load_data(Adult())
    x = data.x[Adult().continuous_features].values
    normalizer = sklearn.preprocessing.Normalizer()
    x_scaled = normalizer.fit_transform(x)
    data.x[Adult().continuous_features] = x_scaled

    train, test = train_test_split(data)

    train_x = pandas.concat([train.x, train.s], axis="columns")
    test_x = pandas.concat([test.x, test.s], axis="columns")

    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_x.values), torch.Tensor(train.y.values))
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_x.values), torch.Tensor(test.y.values))

    return DataSource(train_dataset=train_dataset, test_dataset=test_dataset)


def get_MNIST() -> DataSource:
    # num_classes=10, input_size=28
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("data", train=False, transform=transform)

    return DataSource(train_dataset=train_dataset, test_dataset=test_dataset)


def get_RepeatedMNIST() -> DataSource:
    # num_classes = 10, input_size = 28
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    org_train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_dataset: Dataset = data.ConcatDataset([org_train_dataset] * 3)

    test_dataset = datasets.MNIST("data", train=False, transform=transform)
    return DataSource(train_dataset=train_dataset, test_dataset=test_dataset)


class DatasetEnum(enum.Enum):
    mnist = "mnist"
    emnist = "emnist"
    emnist_bymerge = "emnist_bymerge"
    repeated_mnist_w_noise = "repeated_mnist_w_noise"
    repeated_mnist_w_noise2 = "repeated_mnist_w_noise2"
    repeated_mnist_w_noise5 = "repeated_mnist_w_noise5"
    mnist_w_noise = "mnist_w_noise"
    cinic10 = "cinic10"
    adult = "adult"

    def get_data_source(self) -> DataSource:
        if self == DatasetEnum.mnist:
            return get_MNIST()
        elif self in (
            DatasetEnum.repeated_mnist_w_noise2,
            DatasetEnum.repeated_mnist_w_noise5,
            DatasetEnum.repeated_mnist_w_noise,
            DatasetEnum.mnist_w_noise,
        ):
            # num_classes=10, input_size=28
            num_repetitions = {
                DatasetEnum.mnist_w_noise: 1,
                DatasetEnum.repeated_mnist_w_noise: 3,
                DatasetEnum.repeated_mnist_w_noise2: 2,
                DatasetEnum.repeated_mnist_w_noise5: 5,
            }[self]

            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            org_train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)

            def apply_noise(idx: Tensor, sample: Tensor) -> Tuple[Tensor, Tensor]:
                data, target = sample
                return data + dataset_noise[idx], target

            dataset_noise = torch.empty(
                (len(org_train_dataset) * num_repetitions, 28, 28), dtype=torch.float32
            ).normal_(0.0, 0.1)

            train_dataset = TransformedDataset(
                data.ConcatDataset([org_train_dataset] * num_repetitions), transformer=apply_noise
            )

            test_dataset = datasets.MNIST("data", train=False, transform=transform)

            return DataSource(train_dataset=train_dataset, test_dataset=test_dataset)
        elif self in (DatasetEnum.emnist, DatasetEnum.emnist_bymerge):
            # num_classes=47, input_size=28,
            split = "balanced" if self == DatasetEnum.emnist else "bymerge"
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            train_dataset = datasets.EMNIST("emnist_data", split=split, train=True, download=True, transform=transform)

            test_dataset = datasets.EMNIST("emnist_data", split=split, train=False, transform=transform)

            """
                Table II contains a summary of the EMNIST datasets and
                indicates which classes contain a validation subset in the
                training set. In these datasets, the last portion of the training
                set, equal in size to the testing set, is set aside as a validation
                set. Additionally, this subset is also balanced such that it
                contains an equal number of samples for each task. If the
                validation set is not to be used, then the training set can be
                used as one contiguous set.
            """
            if self == DatasetEnum.emnist:
                # Balanced contains a test set
                split_index = len(train_dataset) - len(test_dataset)
                train_dataset, validation_dataset = src.subrange_dataset.dataset_subset_split(  # type: ignore[assignment]
                    train_dataset, split_index  # type: ignore[arg-type]
                )
            else:
                validation_dataset = None  # type: ignore[assignment]
            return DataSource(
                train_dataset=train_dataset, test_dataset=test_dataset, validation_dataset=validation_dataset
            )
        elif self == DatasetEnum.cinic10:
            return get_CINIC10()
        elif self == DatasetEnum.adult:
            return get_Adult()
        else:
            raise NotImplementedError(f"Unknown dataset {self}!")

    @property
    def num_classes(self) -> int:
        if self in (
            DatasetEnum.mnist,
            DatasetEnum.repeated_mnist_w_noise,
            DatasetEnum.repeated_mnist_w_noise2,
            DatasetEnum.repeated_mnist_w_noise5,
            DatasetEnum.mnist_w_noise,
        ):
            return 10
        elif self in (DatasetEnum.emnist, DatasetEnum.emnist_bymerge):
            return 47
        elif self == DatasetEnum.cinic10:
            return 10
        elif self == DatasetEnum.adult:
            return 2
        else:
            raise NotImplementedError(f"Unknown dataset {self}!")

    def create_bayesian_model(self, device: torch.device) -> Any:
        num_classes = self.num_classes
        if self in (
            DatasetEnum.mnist,
            DatasetEnum.repeated_mnist_w_noise,
            DatasetEnum.repeated_mnist_w_noise2,
            DatasetEnum.repeated_mnist_w_noise5,
            DatasetEnum.mnist_w_noise,
        ):
            return src.models.mnist_model.BayesianNet(num_classes=num_classes).to(device)
        elif self in (DatasetEnum.emnist, DatasetEnum.emnist_bymerge):
            return src.models.emnist_model.BayesianNet(num_classes=num_classes).to(device)
        elif self == DatasetEnum.cinic10:
            return src.models.vgg_model.vgg16_cinic10_bn(pretrained=True, num_classes=num_classes).to(device)
        elif self == DatasetEnum.adult:
            return src.models.adult_model.BayesianNet(num_classes=num_classes, input_dim=102).to(device)
        else:
            raise NotImplementedError(f"Unknown dataset {self}!")

    def create_optimizer(self, model: Any) -> torch.optim.Adam:
        if self == DatasetEnum.cinic10:
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
        else:
            optimizer = optim.Adam(model.parameters())
        return optimizer

    def create_train_model_extra_args(self, optimizer: torch.optim.Adam) -> Dict[Any, Any]:
        return {}

    def train_model(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        validation_loader: DataLoader,
        num_inference_samples: int,
        max_epochs: int,
        early_stopping_patience: int,
        desc: Callable[[str], Callable[[Any], str]],
        log_interval: int,
        device: torch.device,
        epoch_results_store: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, int, Any]:
        model = self.create_bayesian_model(device)
        optimizer = self.create_optimizer(model)
        num_epochs, test_metrics = train_model(
            model,
            optimizer,
            max_epochs,
            early_stopping_patience,
            num_inference_samples,
            test_loader,
            train_loader,
            validation_loader,
            log_interval,
            desc,
            device,
            epoch_results_store=epoch_results_store,
            **self.create_train_model_extra_args(optimizer),
        )
        return model, num_epochs, test_metrics


def get_experiment_data(
    data_source: DataSource,
    num_classes: int,
    initial_samples: Optional[List[int]],
    reduced_dataset: bool,
    samples_per_class: int,
    validation_set_size: int,
    balanced_test_set: Dataset,
    balanced_validation_set: Dataset,
) -> ExperimentData:
    train_dataset, test_dataset, validation_dataset = (
        data_source.train_dataset,
        data_source.test_dataset,
        data_source.validation_dataset,
    )

    active_learning_data = ActiveLearningData(train_dataset)
    if initial_samples is None:
        initial_samples = list(
            itertools.chain.from_iterable(
                get_balanced_sample_indices(
                    get_targets(train_dataset), num_classes=num_classes, n_per_digit=samples_per_class  # type: ignore[arg-type]
                ).values()
            )
        )

    # Split off the validation dataset after acquiring the initial samples.
    active_learning_data.acquire(initial_samples)  # type: ignore[arg-type]

    if validation_dataset is None:
        print("Acquiring validation set from training set.")
        if not validation_set_size:
            validation_set_size = len(test_dataset)  # type: ignore[arg-type]

        if not balanced_validation_set:
            validation_dataset = active_learning_data.extract_dataset(validation_set_size)
        else:
            print("Using a balanced validation set")
            validation_dataset = active_learning_data.extract_dataset_from_indices(  # type: ignore[arg-type]
                balance_dataset_by_repeating(  # type: ignore[arg-type]
                    active_learning_data.available_dataset, num_classes, validation_set_size, upsample=False
                )
            )
    else:
        if validation_set_size == 0:
            print("Using provided validation set.")
            validation_set_size = len(validation_dataset)
        if validation_set_size < len(validation_dataset):
            print("Shrinking provided validation set.")
            if not balanced_validation_set:
                validation_dataset = data.Subset(
                    validation_dataset, torch.randperm(len(validation_dataset))[:validation_set_size].tolist()
                )
            else:
                print("Using a balanced validation set")
                validation_dataset = data.Subset(
                    validation_dataset,
                    balance_dataset_by_repeating(validation_dataset, num_classes, validation_set_size, upsample=False),
                )

    if balanced_test_set:
        print("Using a balanced test set")
        print("Distribution of original test set classes:")
        classes = get_target_bins(test_dataset)  # type: ignore[arg-type]
        print(classes)

        test_dataset = data.Subset(
            test_dataset, balance_dataset_by_repeating(test_dataset, num_classes, len(test_dataset))  # type: ignore[arg-type]
        )

    if reduced_dataset:
        # Let's assume we won't use more than 1000 elements for our validation set.
        active_learning_data.extract_dataset(len(train_dataset) - max(len(train_dataset) // 20, 5000))
        test_dataset = src.subrange_dataset.SubrangeDataset(test_dataset, 0, max(len(test_dataset) // 10, 5000))  # type: ignore[arg-type]
        if validation_dataset:
            validation_dataset = src.subrange_dataset.SubrangeDataset(
                validation_dataset, 0, len(validation_dataset) // 10
            )
        print("USING REDUCED DATASET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    show_class_frequencies = True
    if show_class_frequencies:
        print("Distribution of training set classes:")
        classes = get_target_bins(train_dataset)
        print(classes)

        print("Distribution of validation set classes:")
        classes = get_target_bins(validation_dataset)  # type: ignore[arg-type]
        print(classes)

        print("Distribution of test set classes:")
        classes = get_target_bins(test_dataset)  # type: ignore[arg-type]
        print(classes)

        print("Distribution of pool classes:")
        classes = get_target_bins(active_learning_data.available_dataset)
        print(classes)

        print("Distribution of active set classes:")
        classes = get_target_bins(active_learning_data.active_dataset)
        print(classes)

    print(f"Dataset info:")
    print(f"\t{len(active_learning_data.active_dataset)} active samples")
    print(f"\t{len(active_learning_data.available_dataset)} available samples")
    print(f"\t{len(validation_dataset)} validation samples")  # type: ignore[arg-type]
    print(f"\t{len(test_dataset)} test samples")  # type: ignore[arg-type]

    if data_source.shared_transform is not None or data_source.train_transform is not None:
        train_dataset = TransformedDataset(
            active_learning_data.active_dataset,
            vision_transformer=compose_transformers([data_source.train_transform, data_source.shared_transform]),  # type: ignore[arg-type]
        )
    else:
        train_dataset = active_learning_data.active_dataset

    if data_source.shared_transform is not None or data_source.scoring_transform is not None:
        available_dataset = TransformedDataset(
            active_learning_data.available_dataset,
            vision_transformer=compose_transformers([data_source.scoring_transform, data_source.shared_transform]),  # type: ignore[arg-type]
        )
    else:
        available_dataset = active_learning_data.available_dataset  # type: ignore[assignment]

    if data_source.shared_transform is not None:
        test_dataset = TransformedDataset(test_dataset, vision_transformer=data_source.shared_transform)  # type: ignore[arg-type]
        validation_dataset = TransformedDataset(validation_dataset, vision_transformer=data_source.shared_transform)

    return ExperimentData(
        active_learning_data=active_learning_data,
        train_dataset=train_dataset,  # type: ignore[arg-type]
        available_dataset=available_dataset,
        validation_dataset=validation_dataset,  # type: ignore[arg-type]
        test_dataset=test_dataset,  # type: ignore[arg-type]
        initial_samples=initial_samples,
    )


def compose_transformers(iterable: Iterator) -> Optional[Compose]:
    iterable = list(filter(None, iterable))  # type: ignore[var-annotated]
    if len(iterable) == 0:
        return None
    if len(iterable) == 1:
        return iterable[0]
    return transforms.Compose(iterable)


# TODO: move to utils?
def get_target_bins(dataset: Dataset) -> Counter[int]:
    classes = collections.Counter(int(target) for target in get_targets(dataset))
    return classes


# TODO: move to utils?
def balance_dataset_by_repeating(
    dataset: Dataset, num_classes: int, target_size: int, upsample: bool = True
) -> List[Any]:
    balanced_samples_indices = get_balanced_sample_indices(get_targets(dataset), num_classes, len(dataset)).values()  # type: ignore[arg-type]

    if upsample:
        num_samples_per_class = max(
            max(len(samples_per_class) for samples_per_class in balanced_samples_indices), target_size // num_classes
        )
    else:
        num_samples_per_class = min(
            max(len(samples_per_class) for samples_per_class in balanced_samples_indices), target_size // num_classes
        )

    def sample_indices(indices: Tensor, total_length: int) -> List[int]:
        return (torch.randperm(total_length) % len(indices)).tolist()

    balanced_samples_indices = list(  # type: ignore[assignment]
        itertools.chain.from_iterable(
            [
                [samples_per_class[i] for i in sample_indices(samples_per_class, num_samples_per_class)]  # type: ignore[arg-type]
                for samples_per_class in balanced_samples_indices
            ]
        )
    )

    print(f"Resampled dataset ({len(dataset)} samples) to a balanced set of {len(balanced_samples_indices)} samples!")

    return balanced_samples_indices  # type: ignore[return-value]


# TODO: move to utils?
def get_targets(dataset: Dataset) -> Tensor:
    """Get the targets of a dataset without any target target transforms(!)."""
    if isinstance(dataset, TransformedDataset):
        return get_targets(dataset.dataset)
    if isinstance(dataset, data.Subset):
        targets = get_targets(dataset.dataset)
        return torch.as_tensor(targets)[dataset.indices]  # type: ignore[index]
    if isinstance(dataset, data.ConcatDataset):
        return torch.cat([get_targets(sub_dataset) for sub_dataset in dataset.datasets])
    if isinstance(dataset, TensorDataset):
        return dataset.tensors[1]

    if isinstance(dataset, (datasets.MNIST, datasets.ImageFolder,)):
        return torch.as_tensor(dataset.targets)
    if isinstance(dataset, datasets.SVHN):
        return dataset.labels

    raise NotImplementedError(f"Unknown dataset {dataset}!")
