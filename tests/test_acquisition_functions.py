import itertools

import pytest
import torch
import torch.utils.data
from torchvision import datasets, transforms

import src.acquisition_functions
import src.models.mnist_model
import src.sampler_model

# NOTE: we could replace this with a custom dataset if it becomes a problem on Jekyll.


def test_random_acquistion_function():
    batch_size = 10

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    estimator = src.acquisition_functions.AcquisitionFunction("random")

    scores = torch.tensor([])

    num_iters = 5
    for data, _ in itertools.islice(test_loader, num_iters):
        output = estimator.scorer(data)
        scores = torch.cat((scores, output), dim=0)

    assert scores.shape == (batch_size * num_iters,)


@pytest.mark.parametrize("acquisition_function", src.acquisition_functions.AcquisitionFunction)
def test_acquisition_functions(acquisition_function: src.acquisition_functions.AcquisitionFunction):
    batch_size = 13

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    bayesian_net = src.models.mnist_model.BayesianNet(10)

    estimator = acquisition_function

    scores = torch.tensor([], dtype=torch.float32)

    num_iters = 5
    for data, _ in itertools.islice(test_loader, num_iters):
        model_out = bayesian_net(data, k=1)
        output = estimator.scorer(model_out)
        scores = torch.cat((scores, output.float()), dim=0)

    assert scores.shape == (batch_size * num_iters,)


@pytest.mark.parametrize("af_type", src.acquisition_functions.AcquisitionFunction)
def test_check_input_permutation(af_type: src.acquisition_functions.AcquisitionFunction):

    batch_size = 12

    test_data = torch.rand((batch_size, 3, 10))

    mixture_a = test_data[::2, :]
    mixture_b = test_data[1::2, :]
    mixture_c = test_data

    class Forwarder(torch.nn.Module):
        def forward(self, batch):
            return batch

    forwarder = Forwarder()
    estimator = af_type

    torch.testing.assert_allclose(  # type: ignore[arg-type,attr-defined]
        torch.cat([mixture_a, mixture_b], dim=0), torch.cat([mixture_c[::2], mixture_c[1::2]], dim=0),
    )

    output_a = estimator.compute_scores(forwarder(mixture_a), torch.device("cpu"))
    output_b = estimator.compute_scores(forwarder(mixture_b), torch.device("cpu"))
    output_c = estimator.compute_scores(forwarder(mixture_c), torch.device("cpu"))

    assert len(output_a) == mixture_a.shape[0]
    assert len(output_b) == mixture_b.shape[0]
    assert len(output_c) == mixture_c.shape[0]
    assert len(output_a) + len(output_b) == len(output_c)

    if af_type == src.acquisition_functions.AcquisitionFunction.random:
        return

    torch.testing.assert_allclose(  # type: ignore[attr-defined]
        torch.cat([output_a, output_b], dim=0), torch.cat([output_c[::2], output_c[1::2]], dim=0)
    )
