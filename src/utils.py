"""Utils for local resolution estimation."""

import numpy as np
import torch


def next_fft_size(n: int) -> int:
    """Find the next integer >= n that is a product of only 2, 3, and 5."""
    if n <= 1:
        return 1
    # Generate all 5-smooth numbers up to a safe upper bound
    # (next 5-smooth is at most ~2*n for reasonable n)
    limit = max(2 * n, 32)
    smooth = set()
    for i in range(int(np.log2(limit)) + 1):
        for j in range(int(np.log(limit) / np.log(3)) + 1):
            for k in range(int(np.log(limit) / np.log(5)) + 1):
                val = (2**i) * (3**j) * (5**k)
                if val >= n:
                    smooth.add(val)
    return int(min(smooth))


def pad_tensor(t: torch.Tensor, target_shape: tuple, mirror_fill: bool) -> torch.Tensor:
    """Pad a tensor to a target shape, centered, with optional mirror fill.
    Parameters
    ----------
    t : torch.Tensor
        Input tensor to pad.
    target_shape : tuple
        Desired output shape.
    mirror_fill : bool
        If True, fill padding with mirror (reflect) padding.
        If False, fill with zeros.
    Returns
    -------
    result : torch.Tensor
        Padded tensor of shape target_shape.
    """
    if mirror_fill:
        ndim = t.ndim
        # F.pad reflect needs: 4D for 2D pad, 5D for 3D pad
        # Add leading dims so we always have enough
        x = t
        dims_to_add = max(0, 4 - ndim) if ndim <= 2 else max(0, 5 - ndim)
        for _ in range(dims_to_add):
            x = x.unsqueeze(0)

        pad_widths = []
        for s, ts in reversed(list(zip(t.shape, target_shape, strict=False))):
            total_pad = ts - s
            before = total_pad // 2
            after = total_pad - before
            pad_widths.extend([before, after])

        result = torch.nn.functional.pad(x, pad_widths, mode="reflect")

        # Remove the extra leading dims
        for _ in range(dims_to_add):
            result = result.squeeze(0)
    else:
        result = torch.zeros(target_shape, dtype=t.dtype, device=t.device)
        slices = tuple(
            slice((ts - s) // 2, (ts - s) // 2 + s)
            for s, ts in zip(t.shape, target_shape, strict=False)
        )
        result[slices] = t
    return result


def prepare_halfmaps_for_fft(
    map1: torch.Tensor,
    map2: torch.Tensor,
    pad: int = 0,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, tuple, tuple]:
    """Mirror pad, zero pad to efficient FFT size and calculate FFt.

    Parameters
    ----------
    map1 : torch.Tensor
        First half-map as a float32 2D or 3D tensor.
    map2 : torch.Tensor
        Second half-map as a float32 2D or 3D tensor.
    pad : int
        Number of voxels to mirror-pad on each side.
        This is the edge-padding.
    device : str
        Device for computation (e.g. 'cpu' or 'cuda').

    Returns
    -------
    fft1 : torch.Tensor
        Complex rfftn output of first half-map.
    fft2 : torch.Tensor
        Complex rfftn output of second half-map.
    fft_shape : tuple
        Shape of the padded FFT.
    fft_crop : tuple of slices
        Slices to crop back to original size.
    """
    pad = int(pad)
    assert map1.shape == map2.shape, (
        f"Half-map shapes must match: {map1.shape} vs {map2.shape}"
    )
    assert map1.ndim in (2, 3), f"Expected 2D or 3D maps, got shape {map1.shape}"

    map1 = map1.to(dtype=torch.float32, device=device)
    map2 = map2.to(dtype=torch.float32, device=device)

    padded_shape = tuple(s + 2 * pad for s in map1.shape)
    fft_shape = tuple(next_fft_size(s) for s in padded_shape)
    fft_crop = tuple(
        slice((fft_s - user_s) // 2, (fft_s - user_s) // 2 + user_s)
        for fft_s, user_s in zip(fft_shape, padded_shape, strict=False)
    )

    # First pad to padded_shape with random values, then to fft_shape with zeros
    map1_padded = pad_tensor(map1, padded_shape, mirror_fill=True)
    map2_padded = pad_tensor(map2, padded_shape, mirror_fill=True)
    map1_padded = pad_tensor(map1_padded, fft_shape, mirror_fill=False)
    map2_padded = pad_tensor(map2_padded, fft_shape, mirror_fill=False)

    fft1 = torch.fft.rfftn(map1_padded, dim=list(range(map1.ndim)))
    fft2 = torch.fft.rfftn(map2_padded, dim=list(range(map2.ndim)))

    return fft1, fft2, fft_shape, fft_crop


def calculate_frequency_map(sizeMap: tuple, device: str = "cpu") -> torch.Tensor:
    """
    Calculate the radial frequency map for a 2D or 3D rfftn-shaped array.

    Parameters
    ----------
    sizeMap : tuple
        Spatial shape of the padded map (2D or 3D).
    device : str
        Device to create the tensor on.

    Returns
    -------
    frequency_map : torch.Tensor with shape matching rfftn output,
                    values are radial frequencies in cycles/pixel.
    """
    if len(sizeMap) == 3:
        freqi = torch.fft.fftfreq(sizeMap[0], 1.0, device=device)
        freqj = torch.fft.fftfreq(sizeMap[1], 1.0, device=device)
        freqk = torch.fft.rfftfreq(sizeMap[2], 1.0, device=device)
        fi, fj, fk = torch.meshgrid(freqi, freqj, freqk, indexing="ij")
        frequencyMap = torch.sqrt(fi * fi + fj * fj + fk * fk)
    elif len(sizeMap) == 2:
        freqi = torch.fft.fftfreq(sizeMap[0], 1.0, device=device)
        freqj = torch.fft.rfftfreq(sizeMap[1], 1.0, device=device)
        fi, fj = torch.meshgrid(freqi, freqj, indexing="ij")
        frequencyMap = torch.sqrt(fi * fi + fj * fj)
    return frequencyMap


def calculate_shells(
    apix: float,
    spatial_frequencies: np.ndarray,
    spacing_filter: float,
) -> np.ndarray:
    """Calculate bandpass filter shells for a list of spatial frequencies.

    Parameters
    ----------
    apix : float
        Pixel/voxel size in Angstrom.
    spatial_frequencies : np.ndarray
        1D array of spatial frequencies in 1/Angstrom (i.e. 1/resolution).
    spacing_filter : float
        Fractional shell half-width relative to Nyquist.

    Returns
    -------
    shells : np.ndarray
        Array of shape (N, 2), columns are [low_cutoff, high_cutoff] in 1/Angstrom.
    """
    nyq = 0.5
    norm_freqs = spatial_frequencies * apix
    low = (norm_freqs - nyq * spacing_filter) / apix
    high = (norm_freqs + nyq * spacing_filter) / apix
    return np.stack([low, high], axis=1)


def make_hyptan_bandpass(
    distance: torch.Tensor,
    low_freq: float,
    high_freq: float,
    falloff: float,
) -> torch.Tensor:
    """Construct a hyperbolic tangent bandpass filter.

    Parameters
    ----------
    distance : torch.Tensor
        Frequency distance array (same shape as rfftn output).
    low_freq : float
        Low frequency cutoff.
    high_freq : float
        High frequency cutoff.
    falloff : float
        Controls edge steepness.

    Returns
    -------
    bandpass : torch.Tensor
        Normalised to [0, 1], same shape as distance.
    """
    span = high_freq - low_freq
    pi = torch.pi
    bandpass = 0.5 * (
        torch.tanh(pi * (distance + high_freq) / (falloff * span))
        - torch.tanh(pi * (distance - high_freq) / (falloff * span))
        - torch.tanh(pi * (distance + low_freq) / (falloff * span))
        + torch.tanh(pi * (distance - low_freq) / (falloff * span))
    )
    bandpass /= bandpass.max()
    return bandpass


def apply_bandpass_and_invert(
    fft_map: torch.Tensor,
    bandpass: torch.Tensor,
    fft_shape: tuple,
    fft_crop: tuple,
) -> torch.Tensor:
    """Apply a bandpass filter to an rfftn-transformed map and inverse-transform.

    Parameters
    ----------
    fft_map : torch.Tensor
        Complex tensor, output of torch.fft.rfftn.
    bandpass : torch.Tensor
        Bandpass filter.
    fft_shape : tuple
        Full padded shape for irfftn.
    fft_crop : tuple
        Slices to crop irfftn output back to radius-padded shape.
    device : str
        Torch device string.

    Returns
    -------
    result : torch.Tensor
        Real-space tensor cropped to radius-padded shape.
    """
    ndim = fft_map.ndim
    bp_rfft = bandpass[..., : fft_map.shape[-1]]
    filtered = fft_map * bp_rfft
    result = torch.fft.irfftn(filtered, s=fft_shape, dim=list(range(ndim)))
    return result[fft_crop]
