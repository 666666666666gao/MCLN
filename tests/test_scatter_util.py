import pytest
import torch

from utils import scatter_util


def _float64_reference_mean(src, index, dim_size):
    src = src.detach().cpu().double()
    index = index.detach().cpu()
    result = torch.zeros(
        (dim_size,) + tuple(src.shape[1:]), dtype=torch.float64
    )
    result.index_add_(0, index, src)
    count = torch.bincount(index, minlength=dim_size).double()
    count = count.reshape((dim_size,) + (1,) * (src.dim() - 1))
    return result / count.clamp_min(1)


def _production_shape_inputs(offset):
    generator = torch.Generator(device="cpu").manual_seed(20260716)
    src = torch.rand(50000, 3, generator=generator) * 10.0 + offset
    index = torch.randint(0, 1200, (50000,), generator=generator)
    return src, index


def test_deterministic_scatter_mean_dim0_handles_unsorted_and_missing_labels():
    src = torch.tensor(
        [
            [2.0, 20.0],
            [1.0, 10.0],
            [6.0, 60.0],
            [4.0, 40.0],
            [3.0, 30.0],
        ],
        dtype=torch.float64,
    )
    index = torch.tensor([2, 0, 2, 0, 2], dtype=torch.long)

    actual = scatter_util.deterministic_scatter_mean_dim0(
        src, index, dim_size=4
    )

    expected = torch.tensor(
        [
            [2.5, 25.0],
            [0.0, 0.0],
            [11.0 / 3.0, 110.0 / 3.0],
            [0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-14)
    assert actual.dtype == src.dtype
    assert actual.device == src.device


def test_deterministic_scatter_mean_dim0_infers_size_and_preserves_trailing_shape():
    src = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    index = torch.tensor([3, 1, 3, 1], dtype=torch.long)

    actual = scatter_util.deterministic_scatter_mean_dim0(src, index)

    assert actual.shape == (4, 2, 3)
    torch.testing.assert_close(actual[1], (src[1] + src[3]) / 2)
    torch.testing.assert_close(actual[3], (src[0] + src[2]) / 2)
    assert torch.count_nonzero(actual[0]) == 0
    assert torch.count_nonzero(actual[2]) == 0


@pytest.mark.parametrize("dim_size", [None, 3])
def test_deterministic_scatter_mean_dim0_handles_empty_input(dim_size):
    src = torch.empty((0, 2), dtype=torch.float32)
    index = torch.empty((0,), dtype=torch.long)

    actual = scatter_util.deterministic_scatter_mean_dim0(
        src, index, dim_size=dim_size
    )

    expected_size = 0 if dim_size is None else dim_size
    assert actual.shape == (expected_size, 2)
    assert torch.count_nonzero(actual) == 0


def test_deterministic_scatter_mean_dim0_preserves_gradients():
    src = torch.tensor([1.0, 3.0, 8.0], requires_grad=True)
    index = torch.tensor([0, 0, 2], dtype=torch.long)

    result = scatter_util.deterministic_scatter_mean_dim0(
        src, index, dim_size=3
    )
    result.sum().backward()

    torch.testing.assert_close(
        src.grad, torch.tensor([0.5, 0.5, 1.0]), rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("dim_size", [None, 3])
def test_deterministic_scatter_mean_dim0_empty_input_preserves_gradients(
    dim_size,
):
    src = torch.empty((0, 2), dtype=torch.float32, requires_grad=True)
    index = torch.empty((0,), dtype=torch.long)

    result = scatter_util.deterministic_scatter_mean_dim0(
        src, index, dim_size=dim_size
    )

    assert result.requires_grad
    result.sum().backward()
    assert src.grad is not None
    assert src.grad.shape == src.shape
    assert src.grad.numel() == 0


@pytest.mark.parametrize(
    ("src", "expected_second"),
    [
        (torch.tensor([float("nan"), 2.0]), 2.0),
        (torch.tensor([float("inf"), 2.0]), 2.0),
        (torch.tensor([1.0e30, 1.0]), 1.0),
    ],
    ids=["nan", "inf", "large-finite"],
)
def test_deterministic_scatter_mean_dim0_does_not_contaminate_later_groups(
    src, expected_second,
):
    index = torch.tensor([0, 1], dtype=torch.long)

    actual = scatter_util.deterministic_scatter_mean_dim0(src, index)

    assert actual[1].item() == expected_second


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_deterministic_scatter_mean_dim0_supports_low_precision_cpu(dtype):
    src = torch.tensor([[1.0, 3.0], [3.0, 5.0]], dtype=dtype)
    index = torch.tensor([1, 1], dtype=torch.long)

    actual = scatter_util.deterministic_scatter_mean_dim0(src, index)

    expected = torch.tensor([[0.0, 0.0], [2.0, 4.0]], dtype=dtype)
    assert actual.dtype == dtype
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("offset", [0.0, 10.0, 1000.0])
def test_deterministic_scatter_mean_dim0_avoids_production_shape_cancellation(
    offset,
):
    src, index = _production_shape_inputs(offset)
    expected = _float64_reference_mean(src, index, dim_size=1200)

    actual = scatter_util.deterministic_scatter_mean_dim0(
        src, index, dim_size=1200
    )

    torch.testing.assert_close(
        actual.double(), expected, rtol=0.0, atol=6e-5
    )


@pytest.mark.parametrize(
    ("src", "index", "dim_size", "exception", "message"),
    [
        (1.0, torch.tensor([0]), None, TypeError, "src must be a tensor"),
        (
            torch.tensor([1.0]),
            [0],
            None,
            TypeError,
            "index must be a tensor",
        ),
        (
            torch.tensor(1.0),
            torch.empty(0, dtype=torch.long),
            None,
            ValueError,
            "src must be at least 1-D",
        ),
        (
            torch.tensor([1]),
            torch.tensor([0]),
            None,
            TypeError,
            "src must have a floating-point dtype",
        ),
        (
            torch.tensor([1.0]),
            torch.tensor([[0]]),
            None,
            ValueError,
            "index must be 1-D",
        ),
        (
            torch.tensor([1.0]),
            torch.tensor([0], dtype=torch.int32),
            None,
            TypeError,
            "index must have dtype torch.long",
        ),
        (
            torch.tensor([1.0, 2.0]),
            torch.tensor([0]),
            None,
            ValueError,
            "same length",
        ),
        (
            torch.tensor([1.0]),
            torch.tensor([-1]),
            None,
            ValueError,
            "non-negative",
        ),
        (
            torch.tensor([1.0]),
            torch.tensor([0]),
            True,
            TypeError,
            "dim_size must be an integer",
        ),
        (
            torch.tensor([1.0]),
            torch.tensor([0]),
            1.5,
            TypeError,
            "dim_size must be an integer",
        ),
        (
            torch.tensor([1.0]),
            torch.tensor([0]),
            -1,
            ValueError,
            "dim_size must be non-negative",
        ),
        (
            torch.tensor([1.0]),
            torch.tensor([2]),
            2,
            ValueError,
            "dim_size is too small",
        ),
    ],
)
def test_deterministic_scatter_mean_dim0_rejects_invalid_inputs(
    src, index, dim_size, exception, message
):
    with pytest.raises(exception, match=message):
        scatter_util.deterministic_scatter_mean_dim0(
            src, index, dim_size=dim_size
        )


@pytest.mark.parametrize("offset", [0.0, 10.0, 1000.0])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_deterministic_scatter_mean_dim0_is_exactly_repeatable_on_cuda(offset):
    src_cpu, index_cpu = _production_shape_inputs(offset)
    src = src_cpu.cuda()
    index = index_cpu.cuda()
    previous_setting = torch.are_deterministic_algorithms_enabled()

    try:
        torch.use_deterministic_algorithms(True)
        first = scatter_util.deterministic_scatter_mean_dim0(
            src, index, dim_size=1200
        )
        second = scatter_util.deterministic_scatter_mean_dim0(
            src, index, dim_size=1200
        )
    finally:
        torch.use_deterministic_algorithms(previous_setting)

    assert torch.equal(first, second)
    expected = _float64_reference_mean(src_cpu, index_cpu, dim_size=1200)
    torch.testing.assert_close(
        first.cpu().double(), expected, rtol=0.0, atol=6e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_deterministic_scatter_mean_dim0_rejects_device_mismatch():
    src = torch.tensor([1.0], device="cuda")
    index = torch.tensor([0], dtype=torch.long, device="cpu")

    with pytest.raises(ValueError, match="same device"):
        scatter_util.deterministic_scatter_mean_dim0(src, index)
