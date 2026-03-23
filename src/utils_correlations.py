"""Correlation calculations for local resolution estimation."""

import numpy as np
import torch


def _create_bool_mask(
    window_size_float: float,
    ndim: int,
) -> tuple[np.ndarray, int]:
    """
    Create a spherical/circular boolean mask for the local window.

    Parameters
    ----------
    window_size_float : float
        The (possibly non-integer) radius for the mask.
    ndim : int
        Number of dimensions (2 or 3).

    Returns
    -------
    bool_array : np.ndarray
        Binary mask of shape (window, ..., window).
    """
    window_size_int = int(np.ceil(window_size_float))
    window = window_size_int * 2 + 1
    center = window // 2

    indices = np.indices([window] * ndim)
    distances = np.linalg.norm(indices - center, axis=0)

    bool_array = np.ones([window] * ndim, dtype=np.float32)
    bool_array[distances > window_size_float] = 0.0

    return bool_array, window_size_int


def _extract_windows_3d(
    padded_map: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Extract masked voxel values from a 3D map at multiple positions.

    Parameters
    ----------
    padded_map : torch.Tensor
        Input 3D map of shape (D, H, W).
    positions : torch.Tensor
        Center positions in padded map, shape (N, 3), dtype long.
    offsets : torch.Tensor
        Relative offsets of valid voxels, shape (M, 3), dtype long.

    Returns
    -------
    values : torch.Tensor
        Extracted values of shape (N, M).
    """
    abs_idx = positions[:, None, :] + offsets[None, :, :]  # (N, M, 3)
    return padded_map[abs_idx[..., 0], abs_idx[..., 1], abs_idx[..., 2]]


def _extract_windows_2d(
    padded_map: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Extract masked pixel values from a 2D map at multiple positions.

    Parameters
    ----------
    padded_map : torch.Tensor
        Input 2D map of shape (H, W).
    positions : torch.Tensor
        Center positions in padded map, shape (N, 2), dtype long.
    offsets : torch.Tensor
        Relative offsets of valid voxels, shape (M, 2), dtype long.

    Returns
    -------
    values : torch.Tensor
        Extracted values of shape (N, M).
    """
    abs_idx = positions[:, None, :] + offsets[None, :, :]  # (N, M, 2)
    return padded_map[abs_idx[..., 0], abs_idx[..., 1]]


def correlationCoefficient(sample1: torch.Tensor, sample2: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector has zero norm or if NaN is encountered.
    Works on both CPU and CUDA tensors.
    """
    numerator = torch.dot(sample1, sample2)
    denominator = torch.linalg.norm(sample1) * torch.linalg.norm(sample2)
    if denominator == 0 or torch.isnan(numerator) or torch.isnan(denominator):
        return 0.0
    return float((numerator / denominator).item())


def initial_permutations(
    maxRadius: np.ndarray,
    windowSize: int,
    paddedHalfMap1_array: list[torch.Tensor],
    paddedHalfMap2_array: list[torch.Tensor],
    bool_array: np.ndarray,
    it_randomMaps: int,
    referenceDistSize: int,
    device: torch.device,
) -> torch.Tensor:
    """Calculate permutation null distribution for local cosine similarity.

    Randomly samples window pairs from the half-maps and computes their
    cosine similarity to build a null distribution for permutation testing.

    Parameters
    ----------
    maxRadius : np.ndarray
        Array of maximum radii per dimension.
    windowSize : int
        Half-width of the sampling window.
    paddedHalfMap1_array : list of torch.Tensor
        Padded half-map 1.
    paddedHalfMap2_array : list of torch.Tensor
        Padded half-map 2 (permuted).
    bool_array : np.ndarray
        Binary mask for selecting voxels within the window.
    it_randomMaps : int
        Number of random permutation maps.
    referenceDistSize : int
        Number of samples in the null distribution.
    device : torch.device
        GPU or CPU device.


    Returns
    -------
    permutedCorCoeffs : torch.Tensor
        1D float tensor of permuted correlation coefficients.
    """
    ndim = len(paddedHalfMap1_array[0].shape)
    paddedHalfMap1 = paddedHalfMap1_array[0]

    # Precompute mask offsets once
    bool_indices_np = np.argwhere(bool_array == 1).astype(np.int64)
    offsets_np = bool_indices_np - np.array([windowSize] * ndim, dtype=np.int64)
    t_offsets = torch.as_tensor(offsets_np, dtype=torch.long, device=device)

    extract_fn = _extract_windows_3d if ndim == 3 else _extract_windows_2d

    # Determine how many samples come from each permuted map
    cycle_length = int(np.ceil(referenceDistSize / it_randomMaps))

    permutedCorCoeffs = torch.empty(
        referenceDistSize, dtype=torch.float32, device=device
    )

    # Generate ALL random positions at once
    # Shape: (referenceDistSize, ndim) for map1 positions
    # Shape: (referenceDistSize, ndim) for map2 positions
    low_bounds = maxRadius[:ndim]
    high_bounds = np.array(
        [paddedHalfMap1.shape[d] - maxRadius[d] - 1 for d in range(ndim)]
    )
    all_indices1 = np.stack(
        [
            np.random.randint(low_bounds[d], high_bounds[d], size=referenceDistSize)
            for d in range(ndim)
        ],
        axis=1,
    )  # (referenceDistSize, ndim)
    all_indices2 = np.stack(
        [
            np.random.randint(low_bounds[d], high_bounds[d], size=referenceDistSize)
            for d in range(ndim)
        ],
        axis=1,
    )  # (referenceDistSize, ndim)

    # Process per-permuted-map in chunks (we need different map2 for each cycle)
    for map_idx in range(it_randomMaps):
        start = map_idx * cycle_length
        end = min(start + cycle_length, referenceDistSize)
        if start >= referenceDistSize:
            break

        paddedHalfMap2 = paddedHalfMap2_array[map_idx]

        positions1 = torch.as_tensor(
            all_indices1[start:end], dtype=torch.long, device=device
        )
        positions2 = torch.as_tensor(
            all_indices2[start:end], dtype=torch.long, device=device
        )

        # Batched window extraction: (batch, M) where M = number of mask voxels
        vals1 = extract_fn(paddedHalfMap1, positions1, t_offsets)
        vals2 = extract_fn(paddedHalfMap2, positions2, t_offsets)

        # Batched cosine similarity
        numerator = (vals1 * vals2).sum(dim=1)
        norm1 = torch.linalg.norm(vals1, dim=1)
        norm2 = torch.linalg.norm(vals2, dim=1)
        denominator = norm1 * norm2
        corr = torch.where(
            denominator > 0,
            numerator / denominator,
            torch.zeros(1, device=device, dtype=vals1.dtype),
        )
        permutedCorCoeffs[start:end] = corr

    return permutedCorCoeffs


def runLocal_torch(
    corrBoxSize: np.ndarray,
    maxRadius: np.ndarray,
    window_size: int,
    paddedHalfMap1: torch.Tensor,
    paddedHalfMap2: torch.Tensor,
    stepSize: np.ndarray,
    permuted_map: torch.Tensor,
    bool_array: np.ndarray,
    result_array: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> None:
    """Compute correlation at each grid position and convert to p-values.

    Uses only the masked voxels, then converts to a p-value via the
    permutation distribution. Result is written in-place to ``result_array``.

    Parameters
    ----------
    corrBoxSize : np.ndarray
        Size of the correlation box (per dimension).
    maxRadius : np.ndarray
        Half-window offset (padding) per dimension.
    window_size : int
        Half-window size (lateral dimensions).
    paddedHalfMap1 : torch.Tensor
        Padded half-map 1 (2D or 3D, float32).
    paddedHalfMap2 : torch.Tensor
        Padded half-map 2 (2D or 3D, float32).
    stepSize : np.ndarray
        Step size per dimension.
    permuted_map : torch.Tensor
        Precomputed FSC values from permutation testing (null distribution).
    bool_array : np.ndarray
        Boolean mask for the local window.
    result_array : torch.Tensor
        Output tensor (modified in-place) for p-values.
    device : torch.device
        Target device (e.g. ``torch.device('cuda:0')`` or ``torch.device('cpu')``).
    batch_size : int
        Number of positions to process simultaneously. Tune for GPU memory.
    """
    ndim = len(paddedHalfMap1.shape)

    # --- Precompute valid mask offsets ---
    bool_indices_np = np.argwhere(bool_array == 1).astype(np.int64)
    offsets_np = bool_indices_np - np.array([window_size] * ndim, dtype=np.int64)
    t_offsets = torch.as_tensor(offsets_np, dtype=torch.long, device=device)

    # --- Build grid of center positions ---
    grid_axes = [
        torch.arange(
            len(range(0, int(corrBoxSize[d]), int(stepSize[d]))), device=device
        )
        for d in range(ndim)
    ]
    grid_meshes = torch.meshgrid(*grid_axes, indexing="ij")
    grid = torch.stack(grid_meshes, dim=-1).reshape(-1, ndim)  # (N centerpoints, ndim)

    for d in range(ndim):
        grid[:, d] = maxRadius[d] + grid[:, d] * stepSize[d]

    # Result indices (for writing back)
    result_axes = [
        torch.arange(
            len(range(0, int(corrBoxSize[d]), int(stepSize[d]))), device=device
        )
        for d in range(ndim)
    ]
    result_meshes = torch.meshgrid(*result_axes, indexing="ij")
    result_grid = torch.stack(result_meshes, dim=-1).reshape(-1, ndim)  # (N, ndim)

    num_positions = grid.shape[0]

    extract_fn = _extract_windows_3d if ndim == 3 else _extract_windows_2d

    # Process in batches
    fsc_all = torch.empty(num_positions, dtype=torch.float32, device=device)

    for start in range(0, num_positions, batch_size):
        end = min(start + batch_size, num_positions)
        batch_positions = grid[start:end].long()

        vals1 = extract_fn(paddedHalfMap1, batch_positions, t_offsets)  # (B, M)
        vals2 = extract_fn(paddedHalfMap2, batch_positions, t_offsets)  # (B, M)

        numerator = (vals1 * vals2).sum(dim=1)
        denom1 = (vals1 * vals1).sum(dim=1)
        denom2 = (vals2 * vals2).sum(dim=1)
        denominator = torch.sqrt(denom1) * torch.sqrt(denom2)

        fsc_all[start:end] = torch.where(
            denominator > 0, numerator / denominator, torch.zeros_like(numerator)
        )

    # Sort the null distribution once
    sorted_permuted, _ = torch.sort(permuted_map.to(dtype=torch.float32))
    permuted_len = float(sorted_permuted.numel())

    # searchsorted gives the index where each fsc value would be inserted
    # trying to keep target array sorted
    insert_indices = torch.searchsorted(sorted_permuted, fsc_all, right=True)
    # Number of permuted values > fsc = total - insert_index
    pvalues = (permuted_len - insert_indices.float()) / permuted_len

    # --- Write results back (vectorized for common case) ---
    if ndim == 3:
        result_array[
            result_grid[:, 0].long(),
            result_grid[:, 1].long(),
            result_grid[:, 2].long(),
        ] = pvalues
    elif ndim == 2:
        result_array[
            result_grid[:, 0].long(),
            result_grid[:, 1].long(),
        ] = pvalues


def localResolutionsGPU_torch(
    sample1_filtered: torch.Tensor,
    sample2_filtered: torch.Tensor,
    permutated_sample2_filtered: list[torch.Tensor],
    permutated_sample1_filtered: list[torch.Tensor],
    windowSize: float,
    corrected_box_size: np.ndarray,
    maxRadius: np.ndarray,
    stepSize: np.ndarray,
    it_randomMaps: int,
    referenceDistSize: int,
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Compute local resolution p-values via permutation testing.

    Works on both GPU and CPU transparently.

    Parameters
    ----------
    sample1_filtered : torch.Tensor
        Padded half-map 1 (2D or 3D, float32).
    sample2_filtered : torch.Tensor
        Padded half-map 2 (2D or 3D, float32).
    permutated_sample2_filtered : list of torch.Tensor
        Permuted half-map 2 variants for building the null distribution.
    permutated_sample1_filtered : list of torch.Tensor
        Permuted half-map 1 variants for building the null distribution.
    windowSize : float
        Window radius (float; will be ceiled for integer indexing).
    corrected_box_size : np.ndarray
        Size of the region to process per dimension.
    maxRadius : np.ndarray
        Padding offset per dimension (allowing measurements at edges).
    stepSize : np.ndarray
        Step size per dimension.
    it_randomMaps : int
        Number of random map iterations for permutation testing.
    referenceDistSize : int
        Size of the reference distribution.
    device : torch.device
        GPU or CPU device.
    batch_size : int
        Number of positions to process simultaneously. Tune for GPU memory


    Returns
    -------
    pValueMap : torch.Tensor
        Map of p-values from the permutation test (float16).
    """
    ndim = len(sample1_filtered.shape)

    # --- Create boolean mask ---
    bool_array, windowSize_int = _create_bool_mask(windowSize, ndim)

    # --- Compute permutation null distribution ---
    permuted_map = initial_permutations(
        maxRadius,
        windowSize_int,
        permutated_sample1_filtered,
        permutated_sample2_filtered,
        bool_array,
        it_randomMaps,
        referenceDistSize,
        device,
    )

    # --- Build output shape ---
    output_shape = tuple(
        len(range(maxRadius[i], maxRadius[i] + corrected_box_size[i], stepSize[i]))
        for i in range(ndim)
    )

    # --- Single device (GPU or CPU) ---
    result_array = torch.ones(output_shape, dtype=torch.float32, device=device)

    runLocal_torch(
        corrected_box_size,
        maxRadius,
        windowSize_int,
        sample1_filtered,
        sample2_filtered,
        stepSize,
        permuted_map,
        bool_array,
        result_array,
        device,
        batch_size,
    )

    return result_array.to(dtype=torch.float16)
