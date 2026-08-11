import torch
from typing import Optional, Tuple


_DETERMINISTIC_SCATTER_MAX_PADDED_ELEMENTS = 1 << 20


def _iter_padded_group_chunks(
        counts: Tuple[int, ...], max_padded_rows: int):
    group_start = 0
    row_start = 0
    while group_start < len(counts):
        group_end = group_start
        row_end = row_start
        max_count = 0
        while group_end < len(counts):
            candidate_max = max(max_count, counts[group_end])
            candidate_groups = group_end - group_start + 1
            candidate_rows = candidate_groups * candidate_max
            if group_end > group_start and candidate_rows > max_padded_rows:
                break
            max_count = candidate_max
            row_end += counts[group_end]
            group_end += 1
            if candidate_rows >= max_padded_rows:
                break
        yield group_start, group_end, row_start, row_end, max_count
        group_start = group_end
        row_start = row_end


def deterministic_scatter_mean_dim0(
        src: torch.Tensor,
        index: torch.Tensor,
        dim_size: Optional[int] = None) -> torch.Tensor:
    """Average rows by label without CUDA atomic reductions.

    This deliberately supports only the superpoint-center contract: a floating
    source tensor reduced over dimension zero by a one-dimensional long index.
    Missing labels produce zero rows.
    """
    if not isinstance(src, torch.Tensor):
        raise TypeError("src must be a tensor")
    if not isinstance(index, torch.Tensor):
        raise TypeError("index must be a tensor")
    if src.dim() < 1:
        raise ValueError("src must be at least 1-D")
    if not src.is_floating_point():
        raise TypeError("src must have a floating-point dtype")
    if index.dim() != 1:
        raise ValueError("index must be 1-D")
    if index.dtype != torch.long:
        raise TypeError("index must have dtype torch.long")
    if src.device != index.device:
        raise ValueError("src and index must be on the same device")
    if src.size(0) != index.numel():
        raise ValueError("src and index must have the same length")
    if dim_size is not None:
        if isinstance(dim_size, bool) or not isinstance(dim_size, int):
            raise TypeError("dim_size must be an integer or None")
        if dim_size < 0:
            raise ValueError("dim_size must be non-negative")

    if index.numel() == 0:
        output_size = 0 if dim_size is None else dim_size
        output = src.new_zeros((output_size,) + tuple(src.shape[1:]))
        return output + src.sum() * 0

    sorted_index, order = torch.sort(index, dim=0, stable=True)
    if sorted_index[0].item() < 0:
        raise ValueError("index values must be non-negative")

    required_size = sorted_index[-1].item() + 1
    if dim_size is None:
        dim_size = required_size
    elif dim_size < required_size:
        raise ValueError("dim_size is too small for the largest index")

    accumulation_dtype = (
        torch.float32
        if src.dtype in (torch.float16, torch.bfloat16)
        else torch.float64
    )
    sorted_src = src.index_select(0, order)
    labels, counts = torch.unique_consecutive(
        sorted_index, return_counts=True
    )
    # Reduce independent groups in bounded padded chunks; never subtract
    # cumulative values from a preceding group.
    count_values = tuple(counts.detach().cpu().tolist())
    trailing_elements = 1
    for size in src.shape[1:]:
        trailing_elements *= size
    max_padded_rows = max(
        1,
        _DETERMINISTIC_SCATTER_MAX_PADDED_ELEMENTS
        // max(1, trailing_elements),
    )

    chunk_sums = []
    for (group_start, group_end, row_start, row_end,
         max_count) in _iter_padded_group_chunks(
             count_values, max_padded_rows):
        chunk_counts = counts[group_start:group_end]
        group_count = group_end - group_start
        row_count = row_end - row_start
        group_ids = torch.repeat_interleave(
            torch.arange(
                group_count, dtype=torch.long, device=index.device
            ),
            chunk_counts,
            output_size=row_count,
        )
        group_starts = chunk_counts.cumsum(dim=0) - chunk_counts
        within_group = (
            torch.arange(
                row_count, dtype=torch.long, device=index.device
            )
            - group_starts.index_select(0, group_ids)
        )
        padded_positions = group_ids * max_count + within_group
        padded = src.new_zeros(
            (group_count * max_count,) + tuple(src.shape[1:])
        )
        padded = padded.index_copy(
            0, padded_positions, sorted_src[row_start:row_end]
        )
        padded = padded.reshape(
            (group_count, max_count) + tuple(src.shape[1:])
        )
        chunk_sums.append(
            padded.sum(dim=1, dtype=accumulation_dtype)
        )
    segment_sums = torch.cat(chunk_sums, dim=0)

    count_shape = (counts.numel(),) + (1,) * (src.dim() - 1)
    means = segment_sums / counts.reshape(count_shape).to(
        dtype=accumulation_dtype
    )
    means = means.to(dtype=src.dtype)
    output = src.new_zeros((dim_size,) + tuple(src.shape[1:]))
    return output.index_copy(0, labels, means)


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out
