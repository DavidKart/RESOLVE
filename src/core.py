"""Core functions for local resolution estimation."""

import numpy as np
import torch

from . import utils, utils_correlations


def compute_resolution(
    apix: float,
    windows_radii: np.ndarray,
    resolutions: np.ndarray,
    batch_halfMap1: list[torch.Tensor],
    batch_halfMap2: list[torch.Tensor],
    stepSize: int = 3,
    gpu_ids: int | None = None,
    referenceDistSize: int = 10000,
    phase_permutation: bool = True,
    batch_size: int = 4096,
    shell_size: float = 0.05,
    falloff: float = 1.5,
) -> torch.Tensor:
    """Compute a per-shell p-value map via local correlation, without resolution thresholding.

   Parameters
    ----------
    apix : float
        Voxel size in Ångströms per pixel (Å px⁻¹).
    windows_radii : np.ndarray
        Array of window radii in voxels used for correlation measurements,
        corresponding to the resoluions array.
    resolutions : np.ndarray
        Array of resolutions (Ångströms) to evaluate.
    batch_halfMap1 : list[torch.Tensor]
        List of first half-maps as 2D/3D arrays.
    batch_halfMap2 : list[torch.Tensor]
        List of second half-maps as 2D/3D arrays. Must have the same length
        and element shapes as batch_halfMap1.
    stepSize : int
        Step size in voxels between local correlation sampling positions.
        Larger values reduce computation time but lower spatial sampling.
    gpu_ids : int | None
        GPU device ID to use for computation. None for CPU.
    referenceDistSize : int
        Number of correlation measurements to create reference distribution. Used for
        p-value determination. For small input map size, high referenceDistSize and high
        window radii, multiple maps should are in 'n_randomMaps' to create
        a diverse enough reference distribution (or the reference distribution may
        narrow, leading to unauthentically high p-values.)
    phase_permutation: bool
        Phase permutation instead of real-space permutation for reference distribution
        creation.
    batch_size : int
        Number of positions to process simultaneously.
    shell_size : float
        Controlling the width of the bandpass filter relative to Nyquist.
    falloff : float
        Falloff parameter controlling the sharpness of the bandpass filter.

    Returns
    -------
    localResMap : torch.Tensor
        Map with p-values per batch element, per sampling location, per investigated
        shell, in the order of given input shells. Shape is
        (n_batch, len(windows_radii), *output_shape). Map size will be reduced
        compared to input map according to stepSize.
    """
    n_batch = len(batch_halfMap1)
    assert len(batch_halfMap2) == n_batch, (
        f"batch_halfMap1 and batch_halfMap2 must have the same length, "
        f"got {n_batch} and {len(batch_halfMap2)}"
    )
    assert n_batch > 0, "batch_halfMap1 and batch_halfMap2 must not be empty"
    ref_shape = batch_halfMap1[0].shape
    assert all(m.shape == ref_shape for m in batch_halfMap1), (
        "All elements in batch_halfMap1 must have the same shape"
    )
    assert all(m.shape == ref_shape for m in batch_halfMap2), (
        "All elements in batch_halfMap2 must have the same shape"
    )

   
    if torch.cuda.is_available() and gpu_ids is not None:
        device = torch.device(f"cuda:{gpu_ids}")
    else:
        device = torch.device("cpu")

    # print(f"debug device {device}")

    # pad: required padding for calculations at the edges
    pad = int(np.ceil(np.max(windows_radii)))
    corrected_box_size = ref_shape # original
    dim = len(corrected_box_size) # 2D or 3D
    maxRadius = np.array([pad for _ in range(dim)])
    stepSize_dim = np.array([stepSize for _ in range(dim)])
    output_shape = tuple(len(range(0, s, stepSize)) for s in ref_shape) # output shape reduced by step size
    
    # Estimate required number of random maps
    n_randomMaps = utils.estimate_random_maps(
        referenceDistSize, 
        np.max(windows_radii), 
        ref_shape
        )

    # Per-element FFT preparation and permutation map generation
    fft_pairs = []
    permutation_maps_fft_all = []

    for b in range(n_batch):
        fft1, fft2, fft_shape, fft_crop = utils.prepare_halfmaps_for_fft(
            batch_halfMap1[b],
            batch_halfMap2[b],
            pad=pad,
            device=device,
        )
        fft_pairs.append((fft1.cpu(), fft2.cpu(), fft_shape, fft_crop)) # intermediate cpu transfer for gpu memory relief

        perm_ffts = []
        if phase_permutation:
            fft2_abs = torch.abs(fft2)

        for _ in range(n_randomMaps):
            if phase_permutation:
                angles_flat = torch.angle(fft2).reshape(-1)
                shuffled_angles = angles_flat[
                    torch.randperm(angles_flat.numel(), device=device)
                ].reshape(fft2.shape)
                perm_ffts.append((fft2_abs * torch.exp(1j * shuffled_angles)).cpu())
            else:
                t_flat = batch_halfMap2[b].flatten().float().to(device)
                idx = (
                    torch.randperm(int(np.prod(fft_shape)), device=device)
                    % t_flat.numel()
                )
                permutation_map = t_flat[idx].reshape(fft_shape)
                fft3 = torch.fft.rfftn(permutation_map, dim=list(range(len(fft_shape))))
                perm_ffts.append(fft3.cpu())
        del fft1, fft2

        permutation_maps_fft_all.append(perm_ffts)
        
        # Cleanup
        batch_halfMap1[b] = None
        batch_halfMap2[b] = None

    # Cleanup 
    if phase_permutation:
        del fft2_abs
    else:
        del fft3
    del batch_halfMap1, batch_halfMap2
    torch.cuda.empty_cache()

    # Calculation of frequency map and shells
    frequencyMap = utils.calculate_frequency_map(fft_shape, device) / float(apix)
    resolutions = 1 / resolutions
    shells = utils.calculate_shells(apix, resolutions, shell_size)

    # Preparing empty output map to fill. Dimensions: (n_batch, n_windows, *output_shape)
    locResMap = torch.zeros((n_batch, len(windows_radii), *output_shape), device=device, dtype=torch.float16)   
    
    # Iterate over all windows/resolutions
    for index_i, i in enumerate(windows_radii):
        # print(f"debug: going for resolution {1 / resolutions[index_i]}")
        windowSize = i

        # Bandpass filter creation
        bandpassFilter = utils.make_hyptan_bandpass(
            frequencyMap, shells[index_i][0], shells[index_i][1], falloff
        )

        for b in range(n_batch):
            fft1, fft2, fft_shape_b, fft_crop_b = fft_pairs[b]

            sample1_filtered = utils.apply_bandpass_and_invert(
                fft1.to(device), bandpassFilter, fft_shape_b, fft_crop_b
            )
            sample2_filtered = utils.apply_bandpass_and_invert(
                fft2.to(device), bandpassFilter, fft_shape_b, fft_crop_b
            )

            permutated_sample2_filtered = []
            for ind_rand in range(n_randomMaps):
                permutated_sample2_filtered.append(
                    utils.apply_bandpass_and_invert(
                        permutation_maps_fft_all[b][ind_rand].to(device),
                        bandpassFilter,
                        fft_shape_b,
                        fft_crop_b,
                    )
                )
            permutated_sample1_filtered = [sample1_filtered]

            locResMap[b, index_i] = utils_correlations.localResolutionsGPU_torch(
                sample1_filtered,
                sample2_filtered,
                permutated_sample2_filtered,
                permutated_sample1_filtered,
                windowSize,
                corrected_box_size,
                maxRadius,
                stepSize_dim,
                n_randomMaps,
                referenceDistSize,
                device,
                batch_size,
            )

    return locResMap
