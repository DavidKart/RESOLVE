"""Extra utilities outside of core functionalities."""

import numpy as np
import torch
from scipy.interpolate import interp1d
import json
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_resolutions(
    size_vol: int,
    apix: float,
    res_max: float,
    accuracy_steps: int,
) -> list[float]:
    """Calculate resolution shells for frequency-space filtering.

    Parameters
    ----------
    size_vol : int
        Volume size in voxels (used to compute the real FFT frequency grid).
    apix : float
        Pixel size in Ångströms.
    res_max : float
        Maximum resolution cutoff in Ångströms; shells coarser than this
        are discarded.
    accuracy_steps : int
        Sub-sampling stride – only every *accuracy_steps*-th shell is
        kept (the last shell is always included).

    Returns
    -------
    resolutions : np.ndarray
        resolutions to measure correlation for.
    """
    freq_map = np.fft.rfftfreq(size_vol, 1)
    freq_map = freq_map[1:]  # discard zero component
    res_map = [1 / f for f in freq_map]
    freq_map_apix = freq_map / apix

    resolutions = []
    for idx, res in enumerate(res_map):
        if (idx % accuracy_steps != 0) and (idx != len(res_map) - 1):
            continue
        res_angstrom = 1 / freq_map_apix[idx]
        if res_angstrom > res_max:
            continue
        resolutions.append(res_angstrom)

    return np.array(resolutions)

def get_windows_empirical(
    apix : float,
    input_values: np.ndarray,
    dim: int,
) -> np.ndarray:
    """Look up empirical window radii for each resolution shell.

    Shell-dependent window radii were pre-calculated from simulations
    for 2-D and 3-D measurements and are interpolated here to arbitrary
    query frequencies.

    Parameters
    ----------
    apix : float
        pixel size in Angstrom        
    input_values : np.ndarray
        Resolution values in Angstrom at which to
        evaluate the window-radius curve.
    dim : int
        Dimensionality of the measurement – must be ``2`` or ``3``.

    Returns
    -------
    windows : np.ndarray
        Interpolated window radii, same shape as *input_values*.
    """
    if dim == 2:
        #2DnoBorders_4udf_phaseRand_thresh0.95_molThres0.5_hypTan0.05-falloff1.5.json --- with fsc 0.2
        res = [0.1, 0.10373737373737374, 0.10747474747474747, 0.11121212121212122, 0.11494949494949495, 0.11868686868686869, 0.12242424242424244, 0.12616161616161617, 0.1298989898989899, 0.13363636363636364, 0.13737373737373737, 0.14111111111111113, 0.14484848484848484, 0.1485858585858586, 0.15232323232323233, 0.15606060606060607, 0.1597979797979798, 0.16353535353535353, 0.16727272727272727, 0.171010101010101, 0.17474747474747476, 0.17848484848484847, 0.18222222222222223, 0.18595959595959596, 0.1896969696969697, 0.19343434343434343, 0.19717171717171716, 0.20090909090909093, 0.20464646464646463, 0.2083838383838384, 0.21212121212121213, 0.21585858585858586, 0.2195959595959596, 0.22333333333333333, 0.22707070707070706, 0.2308080808080808, 0.23454545454545453, 0.2382828282828283, 0.24202020202020202, 0.24575757575757576, 0.2494949494949495, 0.25323232323232325, 0.25696969696969696, 0.26070707070707066, 0.2644444444444445, 0.2681818181818182, 0.2719191919191919, 0.27565656565656566, 0.2793939393939394, 0.2831313131313131, 0.28686868686868683, 0.2906060606060606, 0.29434343434343435, 0.29808080808080806, 0.3018181818181818, 0.3055555555555556, 0.3092929292929293, 0.313030303030303, 0.31676767676767675, 0.3205050505050505, 0.3242424242424242, 0.327979797979798, 0.33171717171717174, 0.33545454545454545, 0.33919191919191916, 0.3429292929292929, 0.3466666666666667, 0.3504040404040404, 0.3541414141414141, 0.3578787878787879, 0.3616161616161616, 0.3653535353535353, 0.369090909090909, 0.37282828282828284, 0.37656565656565655, 0.38030303030303025, 0.38404040404040407, 0.3877777777777778, 0.3915151515151515, 0.3952525252525252, 0.398989898989899, 0.4027272727272727, 0.4064646464646464, 0.41020202020202023, 0.41393939393939394, 0.41767676767676765, 0.42141414141414135, 0.42515151515151517, 0.4288888888888889, 0.4326262626262626, 0.4363636363636364, 0.4401010101010101, 0.4438383838383838, 0.4475757575757575, 0.45131313131313133, 0.45505050505050504, 0.45878787878787874, 0.46252525252525256, 0.46626262626262627, 0.47]
        windows = [49.36361869513791, 48.101260190452166, 46.86436042293574, 45.652919392588636, 44.46693709941084, 43.30641354340237, 42.17134872456322, 41.06174264289338, 39.97759529839286, 38.918906691061665, 37.88567682089978, 36.877905687907216, 35.895593292083966, 34.93873963343003, 34.00734471194542, 33.10140852763013, 32.22093108048415, 31.365912370507488, 30.536352397700146, 29.732251162062123, 28.953608663593418, 28.200424902294028, 27.472699878163954, 26.7704335912032, 26.093626041411767, 25.045565672854316, 24.398520370005397, 23.81035338175588, 23.270603830696924, 22.76867693319509, 22.289111082809953, 21.81817466072031, 21.361007203926064, 20.918186988054018, 20.486738822692256, 20.074835893503465, 19.697334857160612, 19.35801431720891, 19.0567630880189, 18.794672840227026, 18.562181792575036, 18.333626980491182, 18.09547602764037, 17.85631235059332, 17.621778528257025, 17.387999403843313, 17.15725532789141, 16.929174803548424, 16.702091760237305, 16.46919359395626, 16.2277514636398, 15.98378734028979, 15.750907875438257, 15.524943890633782, 15.297581550630724, 15.069876673755571, 14.84968026435418, 14.644798630736183, 14.458253889087974, 14.29468246400815, 14.158327090393335, 14.046873235031862, 13.951641169273067, 13.861745772148343, 13.779821423018864, 13.71046984654102, 13.656155260887243, 13.617206251565866, 13.593255117623823, 13.579620305949607, 13.576334502320877, 13.585110757334864, 13.604955215952494, 13.631404968211983, 13.663444789388635, 13.644337010640735, 13.60757937561159, 13.572765637206391, 13.539895795425144, 13.508969850267846, 13.479987801734499, 13.4529496498251, 13.427855394539652, 13.404705035878152, 13.383498573840601, 13.364236008427003, 13.346917339637352, 13.33154256747165, 13.3181116919299, 13.306624713012098, 13.297081630718246, 13.289482445048344, 13.283827156002392, 13.280115763580389, 13.278348267782336, 13.278524668608233, 13.280644966058079, 13.284709160131873, 13.29071725082962, 13.298669238151314]
    elif  dim == 3:
        #3D_4udf_phaseRand_thresh0.95_molThres0.3_hypTan0.05-falloff1.5
        res = [0.1, 0.10373737373737374, 0.10747474747474747, 0.11121212121212122, 0.11494949494949495, 0.11868686868686869, 0.12242424242424244, 0.12616161616161617, 0.1298989898989899, 0.13363636363636364, 0.13737373737373737, 0.14111111111111113, 0.14484848484848484, 0.1485858585858586, 0.15232323232323233, 0.15606060606060607, 0.1597979797979798, 0.16353535353535353, 0.16727272727272727, 0.171010101010101, 0.17474747474747476, 0.17848484848484847, 0.18222222222222223, 0.18595959595959596, 0.1896969696969697, 0.19343434343434343, 0.19717171717171716, 0.20090909090909093, 0.20464646464646463, 0.2083838383838384, 0.21212121212121213, 0.21585858585858586, 0.2195959595959596, 0.22333333333333333, 0.22707070707070706, 0.2308080808080808, 0.23454545454545453, 0.2382828282828283, 0.24202020202020202, 0.24575757575757576, 0.2494949494949495, 0.25323232323232325, 0.25696969696969696, 0.26070707070707066, 0.2644444444444445, 0.2681818181818182, 0.2719191919191919, 0.27565656565656566, 0.2793939393939394, 0.2831313131313131, 0.28686868686868683, 0.2906060606060606, 0.29434343434343435, 0.29808080808080806, 0.3018181818181818, 0.3055555555555556, 0.3092929292929293, 0.313030303030303, 0.31676767676767675, 0.3205050505050505, 0.3242424242424242, 0.327979797979798, 0.33171717171717174, 0.33545454545454545, 0.33919191919191916, 0.3429292929292929, 0.3466666666666667, 0.3504040404040404, 0.3541414141414141, 0.3578787878787879, 0.3616161616161616, 0.3653535353535353, 0.369090909090909, 0.37282828282828284, 0.37656565656565655, 0.38030303030303025, 0.38404040404040407, 0.3877777777777778, 0.3915151515151515, 0.3952525252525252, 0.398989898989899, 0.4027272727272727, 0.4064646464646464, 0.41020202020202023, 0.41393939393939394, 0.41767676767676765, 0.42141414141414135, 0.42515151515151517, 0.4288888888888889, 0.4326262626262626, 0.4363636363636364, 0.4401010101010101, 0.4438383838383838, 0.4475757575757575, 0.45131313131313133, 0.45505050505050504, 0.45878787878787874, 0.46252525252525256, 0.46626262626262627, 0.47]
        windows = [10.427159025265857, 10.224550648987888, 10.024724105932263, 9.827679396098981, 9.633416519488042, 9.441935476099447, 9.253236265933197, 9.067318888989288, 8.884183345267724, 8.703829634768502, 8.526257757491624, 8.35146771343709, 8.1794595026049, 8.010233124995052, 7.843788580607548, 7.680125869442387, 7.51924499149957, 7.3611459467790965, 7.205828735280966, 7.053293357005179, 6.9035398119517355, 6.756568100120637, 6.612378221511879, 6.4709701761254665, 6.332343963961397, 6.196499585019671, 6.063437039300288, 5.933156326803249, 5.805657447528553, 5.6809404014762, 5.559005188646191, 5.439851809038525, 5.323480262653203, 5.209890549490225, 5.09908266954959, 4.991056622831297, 4.885812409335349, 4.783350029061744, 4.683669482010483, 4.586770768181565, 4.49265388757499, 4.401318840190759, 4.312765626029701, 4.2443780015069645, 4.179415873372608, 4.11804602113812, 4.060340891193263, 4.006316016955979, 3.955493167471093, 3.9073774135683905, 3.8615726095125815, 3.8177800341188, 3.7757507604214515, 3.735270064878486, 3.6961391384713274, 3.6581785320472964, 3.621257850052169, 3.585270636114754, 3.5356471752345815, 3.4872090136596845, 3.439956151389373, 3.393888588423646, 3.3490063247625046, 3.3053093604059476, 3.2627976953539757, 3.2214713296065884, 3.181330263163787, 3.14237449602557, 3.1046040281919387, 3.068018859662892, 3.0326189904384298, 2.9984044205185527, 2.965375149903261, 2.9335311785925544, 2.902872506586433, 2.873399133884896, 2.845111060487944, 2.8180082863955773, 2.792090811607795, 2.767358636124599, 2.7438117599459875, 2.72145018307196, 2.7002739055025184, 2.6802829272376627, 2.6614772482773903, 2.643856868621704, 2.6274217882706026, 2.6121720072240855, 2.5981075254821535, 2.5852283430448075, 2.573534459912045, 2.563025876083869, 2.5537025915602776, 2.5455646063412702, 2.538611920426849, 2.532844533817012, 2.5282624465117607, 2.524865658511094, 2.522654169815012, 2.5216279804235153]
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    input_values = (1/np.array(input_values))*apix

    interp_func = interp1d(res, windows, kind='linear', fill_value="extrapolate")
    output = np.array(interp_func(input_values).tolist()) 

    return output 

def interpolate_with_zoom(
    input_map: torch.Tensor,
    output_shape: tuple[int, ...],
    step_size: tuple[int, ...],
    low_res: float,
) -> torch.Tensor:
    """Rescale a map with trilinear interpolation and embed it in the output grid.

    The input is resized so that it covers the strided sample positions
    defined by *step_size*, then placed into a full-sized output tensor
    whose remaining voxels are filled with *low_res*.

    Parameters
    ----------
    input_map : torch.Tensor
        Map to be interpolated (2-D or 3-D).
    output_shape : tuple of int
        Desired shape of the returned tensor.
    step_size : tuple of int
        Sampling stride along each axis – determines how many output
        voxels the zoomed map should span.
    low_res : float
        Fill value for output voxels outside the zoomed region
        (typically the lowest resolution present in the data).

    Returns
    -------
    output_map : torch.Tensor
        Interpolated map with shape *output_shape*, on the same device
        as *input_map*.
    """
    ndim = len(output_shape)
    target_size = [
        int(range(0, output_shape[i], step_size[i])[-1]) + 1
        for i in range(ndim)
    ]

    # F.interpolate expects (N, C, *spatial) layout
    mode = "trilinear" if ndim == 3 else "bilinear"
    zoomed_map = torch.nn.functional.interpolate(
        input_map.unsqueeze(0).unsqueeze(0).float(),
        size=target_size,
        mode=mode,
        align_corners=False,
    ).squeeze(0).squeeze(0).to(input_map.dtype)

    output_map = input_map.new_full(output_shape, low_res)
    slices = tuple(slice(0, min(zoomed_map.shape[i], output_shape[i])) for i in range(ndim))
    output_map[slices] = zoomed_map[slices]

    return output_map


def p_adjust_by(p_values: torch.Tensor) -> torch.Tensor:
    """Benjamini–Yekutieli FDR correction, vectorised over a batch of p-value vectors.

    Applies the BY step-up procedure to every row of *p_values*
    independently, returning the corresponding adjusted q-values.
    The harmonic-number approximation uses the same Euler–Maclaurin
    expansion as the original CPU implementation.

    Parameters
    ----------
    p_values : torch.Tensor
        Tensor of shape ``(batch, num_shells)`` containing raw p-values.

    Returns
    -------
    q_values : torch.Tensor
        Tensor of the same shape as *p_values* with BY-adjusted q-values,
        each row corrected independently.
    """
    batch, num_p = p_values.shape
    n = float(num_p)
    # H_n ≈ ln(n) + γ + 1/(2n) − 1/(12n²) + 1/(120n⁴)
    Hn = math.log(n) + 0.5772 + 0.5 / n - 1.0 / (12.0 * n**2) + 1.0 / (120.0 * n**4)

    p_sort_ind = torch.argsort(p_values, dim=1)
    p_sort = torch.gather(p_values, 1, p_sort_ind)

    ranks = torch.arange(1, num_p + 1, device=p_values.device, dtype=p_values.dtype)
    correction = (n / ranks) * Hn  # (num_p,)

    p_adjusted_raw = p_sort * correction.unsqueeze(0)  # (batch, num_p)

    # Reverse cumulative minimum (right-to-left running min, clamped to 1.0)
    p_adjusted = torch.flip(p_adjusted_raw, dims=[1])
    p_adjusted = torch.cummin(p_adjusted, dim=1).values
    p_adjusted = torch.flip(p_adjusted, dims=[1])
    p_adjusted = torch.clamp(p_adjusted, max=1.0)

    # Unsort to original order
    unsort_ind = torch.argsort(p_sort_ind, dim=1)
    q_values = torch.gather(p_adjusted, 1, unsort_ind)
    return q_values


def calc_res_index(
    q_vals: torch.Tensor,
    p_cutoff: float,
    test2: bool,
) -> torch.Tensor:
    """Vectorised resolution-index calculation from BY-adjusted q-values.

    For every voxel (row) the function walks through the shells in order
    and determines which shell index defines the resolution.

    * **test2 = False** – return the index of the last shell in the
      first contiguous run of q ≤ *p_cutoff* starting from shell 0.
      The walk stops at the first failure.
    * **test2 = True** – the walk tolerates exactly one failing shell:.

    The loop runs over the (small) shell dimension while the large
    voxel dimension stays fully parallel.

    Parameters
    ----------
    q_vals : torch.Tensor
        Shape ``(batch, num_shells)`` of adjusted q-values.
    p_cutoff : float
        Significance threshold.
    test2 : bool
        Whether to allow a single-shell gap (see above).

    Returns
    -------
    res_index : torch.Tensor
        Long tensor of shape ``(batch,)`` – selected shell index per
        voxel, or -1 where no resolution is assigned.
    """
    batch, num_shells = q_vals.shape
    device = q_vals.device

    passes = q_vals <= p_cutoff  # (batch, num_shells)

    res_index = torch.full((batch,), -1, dtype=torch.long, device=device)
    testing = torch.ones(batch, dtype=torch.bool, device=device)
    active = torch.ones(batch, dtype=torch.bool, device=device)

    for x in range(num_shells):
        p = passes[:, x]

        # First branch: q <= p_cutoff AND testing
        branch1 = active & testing & p

        if not test2:
            res_index = torch.where(branch1, x, res_index)
            active = active & branch1
        else:
            res_index = torch.where(branch1, x, res_index)

            # Else branch: active voxels that did NOT hit branch1
            in_else = active & ~branch1

            # First time in else (testing was True) → advance index, set testing=False
            first_gap = in_else & testing
            res_index = torch.where(first_gap, x, res_index)

            # Second time in else (testing was False) → decrement and break
            second_else = in_else & ~testing
            res_index = torch.where(second_else, res_index - 1, res_index)
            active = active & ~second_else

            # All else-branch voxels: testing becomes False
            testing = testing & ~in_else

    return res_index


def fill_map(
    loc_res_map: torch.Tensor,
    resolutions: torch.Tensor,
    p_cutoff: float,
    low_res: float,
    test2: bool,
    chunk_size: int = 2**20,
) -> torch.Tensor:
    """Build a resolution map by applying BY-FDR correction across shells.

    For every voxel collect per-shell p-values, apply
    Benjamini–Yekutieli FDR correction, determine the resolution-shell
    index that passes the *p_cutoff* threshold
    
    Parameters
    ----------
    loc_res_map : torch.Tensor
        Per-shell p-value maps.  Each element has shape
        *local_res_map_size* (2-D or 3-D) and corresponds to one
        resolution shell.  The number of elements equals the number of
        shells tested.
    res : torch.Tensor
        1-D tensor of recirpocal resolution values (one per shell, same length as
        *loc_res_map*).
    p_cutoff : float
        Significance threshold applied *after* BY correction.
    low_res : float
        Lowest resolution to test.
    test2 : bool
        If ``True``, tolerate a single-shell gap when walking through
        the q-value sequence (see :func:`calc_res_index`).
    chunk_size: int 
        Processing in chunks to avoid memory overload.

    Returns
    -------
    output_map : torch.Tensor
        Resolution map of shape *output_shape*, on the same device as
        the input tensors.  Each voxel contains the resolution.
    """
    device = loc_res_map[0].device
    dtype = loc_res_map[0].dtype
    num_shells = len(loc_res_map)
    mapShape = loc_res_map[0].shape

    p_values = loc_res_map.reshape(num_shells, -1).T
    
    # BY FDR correction – batch over all voxels in chunks to avoid memory overflows
    num_voxels = p_values.shape[0]
    output_map = torch.full((num_voxels,), low_res, device=device, dtype=dtype)
    
    for start in range(0, num_voxels, chunk_size):
        end = min(start + chunk_size, num_voxels)
        chunk = p_values[start:end]
        q_chunk = p_adjust_by(chunk)

        # Resolution index per voxel
        res_indices = calc_res_index(q_chunk, p_cutoff, test2)

        # Convert to truncated resolution values
        valid_mask = res_indices >= 0
        if valid_mask.any():
            sel = res_indices[valid_mask]
            sel = resolutions[sel]
            truncated = torch.floor(sel * 100.0) / 100.0
            output_map[start:end][valid_mask] = truncated
            
        del q_chunk, res_indices, chunk
        
    return output_map.reshape(mapShape)


def calculate_median_res(
    loc_res_map: np.ndarray,
    signal_mask_step_size: np.ndarray,
    resolutions: list[float],
    dimension: int,
    config: str,
    mask_measure: str,
) -> tuple[dict[float, list[float]], float]:
    """Compute per-shell summary statistics of p-values inside a signal mask.

    For each resolution shell the function collects all p-values that
    fall within *signal_mask_step_size*, computes either the median or
    the mean (controlled by *mask_measure*), and stores the result in a
    dictionary keyed by the truncated reciprocal resolution.

    When ``dimension == 3`` or ``config == "Tilt-Series"``, each
    z-slice of a shell is summarised independently (one value per slice);
    otherwise the entire shell volume is reduced to a single value.

    Parameters
    ----------
    loc_res_map : np.ndarray
        Per-shell p-value maps. Array whose first axis indexes
        resolution shells, with spatial dimensions matching
        *signal_mask_step_size*.
    signal_mask_step_size : np.ndarray
        Binary mask (values 0 or 1) with the same spatial dimensions as
        each element of *loc_res_map* (or, for the per-slice branch,
        one 2-D slice per z-index).
    resolutions : list of float
        Resolution value associated with each shell (same length as
        the first axis of *loc_res_map*).
    dimension : int
        Spatial dimensionality (2 or 3).
    config : str
        "Refined-Maps", "Micrographs", "Tilt-Series", or "Tomograms".
    mask_measure : str
        ``"median"`` or ``"average"`` – the summary statistic to apply
        to the masked p-values.

    Returns
    -------
    dict_tilt_series : dict[float, list[float]]
        Mapping from truncated resolution to a list of summary
        statistics (one entry per z-slice or per shell, depending on
        the branch taken).
    signal_ratio : float
        Ratio of map where signal is found (as defined via
        *mask_measure* in the input).
    """
    
    # reduce_fn = np.median if mask_measure == "median" else np.mean
    reduce_fn = {
    "mean": np.mean,
    "median": lambda x, **kw: np.quantile(x, 0.50, **kw),
    }[mask_measure]
    
    
    dict_tilt_series: dict[float, list[float]] = {}
    actualValues = 0
    overallValues = np.prod(loc_res_map.shape)
    for index_i in range(len(loc_res_map)):
        res_rounded = int(resolutions[index_i] * 100) / 100

        if res_rounded not in dict_tilt_series:
            dict_tilt_series[res_rounded] = []

        if (config == "Tilt-Series") or (dimension == 3):
            for z_slice in range(loc_res_map[index_i].shape[0]):
                p_values_z = loc_res_map[index_i][z_slice]
                p_values_z = p_values_z[signal_mask_step_size[z_slice] == 1]
                actualValues += p_values_z.size
                if p_values_z.size != 0:
                    val = float(reduce_fn(p_values_z))
                    dict_tilt_series[res_rounded].append(val)
                else:
                    dict_tilt_series[res_rounded].append(1.0)
        else:
            p_values = loc_res_map[index_i]
            p_values = p_values[signal_mask_step_size == 1]
            actualValues += p_values.size
            dict_tilt_series[res_rounded].append(float(reduce_fn(p_values)))
    return dict_tilt_series, round(actualValues / overallValues, 2)


def plot_heatmap_pvalue_median(
    x_values: list[float],
    y_values: list[float],
    output_path: str,
    minV: float,
    maxV: float,
    xAxisLabel: str,
    yAxisLabel: str,
    figSizeX: float = 12,
    figSizeY: float = 4,
    format: str = "png",
    actualResGlobal: float = 0,
    ratioSignal: float = 0,
) -> None:
    """Plot a q-value curve used to derive median resolution.

    The function plots summary statistics (median or mean
    p-values) against resolution and saves the figure to
    disk. This is referred to as the "median p-value plot" in the paper and
    is produced for tomograms, tilt-series, and micrographs.

    Parameters
    ----------
    x_values : list of float
        Resolution values for each shell.
    y_values : list of float
        Corresponding summary statistics (e.g. median p-values) for
        each resolution shell.
    output_path : str
        Destination file path **without** extension.  The appropriate
        suffix is appended based on *format*.
    minV : float
        Lower bound of the y-axis.
    maxV : float
        Upper bound of the y-axis.
    xAxisLabel : str
        Label for the x-axis.
    yAxisLabel : str
        Label for the y-axis.
    figSizeX : float, optional
        Figure width in inches (default 12).
    figSizeY : float, optional
        Figure height in inches (default 4).
    format : str, optional
        Output format – ``"png"`` (default), ``"svg"``, or ``"pdf"``.
    actualResGlobal : float, optional
        Median resolution value displayed in the plot title (default 0).
    ratioSignal : float, optional
        Fraction of voxels inside the signal mask, displayed in the
        plot title (default 0).

    Returns
    -------
    None
        The figure is written to *output_path* with the chosen format
        extension.  No value is returned.
    """

    matplotlib.use("Agg")

    x_values = np.array(x_values, dtype=np.float32) # Resolutions
    x_values = 1/x_values
    y_values = np.array(y_values, dtype=np.float32)  # p-values

    # Create the plot
    plt.figure(figsize=(figSizeX, figSizeY))
    plt.rcParams['font.size'] = 14

    plt.plot(x_values, y_values, linestyle='-', marker='o', color='b', markersize=3)
    plt.xlabel(xAxisLabel)
    plt.ylabel(yAxisLabel + " p-value")
    plt.ylim(minV, maxV)  
    plt.grid(False)
    plt.title(yAxisLabel + " resolution (FDR-corrected) " + str(actualResGlobal) +  " (signal ratio: " + str(ratioSignal) + ")")
    plt.tight_layout()

    # Save the plot in the specified format
    if format == "svg":
        plt.rcParams["svg.fonttype"] = "none"
        plt.savefig(output_path + ".svg", format="svg")
    elif format == "pdf":
        plt.rcParams["svg.fonttype"] = "none"
        plt.savefig(output_path + ".pdf", format="pdf")
    else:
        plt.savefig(output_path + ".png")
    plt.close()

def plot_heatmap_pvalue(
    data_dict: dict[float | str, list[float]],
    output_path: str,
    minV: float,
    maxV: float,
    xAxisLabel: str,
    yAxisLabel: str,
    cMapLabel: str,
    figSizeX: float = 10,
    figSizeY: float = 4,
    format: str = "png",
    actualResGlobal: float = 0,
    ratioSignal: float = 0,
) -> None:
    """Plot a 2-D p-value heatmap used to derive median resolution.

    The function builds a 2-D array from *data_dict*, where each key
    represents a resolution shell and each value is a list of per-slice
    summary statistics.  This is referred to as the
    "p-value plot" in the paper and is produced for tomograms and
    tilt-series.

    Parameters
    ----------
    data_dict : dict of {float | str : list of float}
        Mapping from resolution (or its string representation) to a
        list of per-slice summary statistics.  Each list becomes one
        row of the heatmap.
    output_path : str
        Destination file path.
    minV : float
        Lower bound of the colour-map range.
    maxV : float
        Upper bound of the colour-map range.
    xAxisLabel : str
        Label for the x-axis.
    yAxisLabel : str
        Label for the y-axis.
    cMapLabel : str
        Label for the colour-bar.
    figSizeX : float, optional
        Figure width in inches (default 10).
    figSizeY : float, optional
        Figure height in inches (default 4).
    format : str, optional
        Output format – ``"png"`` (default), ``"svg"``, or ``"pdf"``.
    actualResGlobal : float, optional
        Median resolution value displayed in the plot title (default 0).
    ratioSignal : float, optional
        Fraction of voxels inside the signal mask, displayed in the
        plot title.

    Returns
    -------
    None
        The figure is written to *output_path* with the chosen format
        extension.  No value is returned.
    """
    matplotlib.use("Agg")

    # Create 2d array from heatmap
    if len(list(data_dict.keys())) > 1:
        first_key = next(iter(data_dict))
        if isinstance(first_key, str):
            sorted_keys = sorted(np.array(list(data_dict.keys())).astype(np.float32))
            sorted_values = [data_dict[str(key)] for key in sorted_keys]
        else:
            sorted_keys = sorted(data_dict.keys())
            sorted_values = [data_dict[key] for key in sorted_keys]
        sorted_keys = np.round(np.array(sorted_keys), 1)
    else:
        sorted_keys = sorted(data_dict.keys())
        sorted_values = [data_dict[key] for key in sorted_keys]


    heatmap_data = np.array(sorted_values)

    plt.figure(figsize=(figSizeX, figSizeY))
    plt.rcParams['font.size'] = 16
    ax = sns.heatmap(
        heatmap_data, 
        cmap="RdBu_r", 
        vmin=minV, 
        vmax=maxV, 
        xticklabels=True, 
        yticklabels=sorted_keys
    )

    ax.set_xlabel(xAxisLabel)
    ax.set_ylabel(yAxisLabel)

    cbar = ax.collections[0].colorbar  
    num_colorbar_ticks = int(figSizeY * 1.5)  
    cbar_ticks = np.linspace(minV, maxV, num_colorbar_ticks) 
    cbar.set_ticks(cbar_ticks) 
    cbar.set_label(cMapLabel) 
    cbar.set_label(cMapLabel, size=16)
    ax.collections[0].colorbar.ax.invert_yaxis()

    #x-axis ticks
    num_xticks = min(10, heatmap_data.shape[1]) 
    x_ticks_positions = np.linspace(0, heatmap_data.shape[1] - 1, num_xticks)
    ax.set_xticks(x_ticks_positions)
    ax.set_xticklabels([f"{int(tick):d}" for tick in x_ticks_positions])
    ax.tick_params(axis='x', labelrotation=0)

    #y-axis ticks
    num_yticks = min(10, heatmap_data.shape[0]) 
    y_ticks_positions = np.linspace(0, heatmap_data.shape[0] - 1, num_yticks)
    ax.set_yticks(y_ticks_positions)
    ax.set_yticklabels([sorted_keys[int(tick)] for tick in y_ticks_positions])

    # if (actualResGlobal != 0) and (ratioSignal != 0):
    plt.title("signal ratio " + str(ratioSignal) + "\nmedian resolution within signal " + str(actualResGlobal))
    plt.tight_layout()
    if format == "svg":
        plt.rcParams["svg.fonttype"] = "none"
        plt.savefig(output_path + ".svg", format="svg")
    elif format == "pdf":
        plt.rcParams["svg.fonttype"] = "none"
        plt.savefig(output_path + ".pdf", format="pdf")
    else:
        plt.savefig(output_path + ".png")
    plt.close()


def getFittedResolution(
    x_list: list[float],
    y_list: list[float],
    lowResRounded: float,
    p_cutoff: float = 0.05,
    num_samples: int = 100,
) -> tuple[float, float]:
    """Determine resolution cutoff from an interpolated FSC curve with FDR correction.

    The function linearly interpolates the p-value
    curve onto a regular grid and applies Benjamini–Yekutieli 
    FDR correction to the p-values.

    Parameters
    ----------
    x_list : list of float
        Resolution values for each shell (x-coordinates).
    y_list : list of float
        Corresponding p-values for each shell (y-coordinates).
    lowResRounded : float
        Fallback resolution returned when no shell passes the FDR
        threshold.
    p_cutoff : float, optional
        Significance threshold applied to the FDR-corrected q-values
        (default 0.05).
    num_samples : int, optional
        Number of evenly spaced points used for interpolation
        (default 100).

    Returns
    -------
    mean_complement : float
        One minus the mean of the FDR-corrected q-values, serving as
        an overall confidence score for the curve.
    actualRes_global : float
        Estimated resolution cutoff (truncated to two decimal places),
        or *lowResRounded* if no shell passes the significance
        threshold.
    """
    
    # Ensure x and y are numpy arrays
    x = np.array(x_list)
    y = np.array(y_list)

    # Create interpolation function
    interp_func = interp1d(x, y, kind='linear', fill_value='extrapolate')

    # Generate regular x-values for interpolation (num_samples)
    x_min = np.min(x)
    x_max = np.max(x)
    sampled_x = np.linspace(x_min, x_max, num_samples)

    # Interpolate y-values at regular x intervals
    interpolated_y = interp_func(sampled_x)
    tensorBatch = torch.tensor(np.array([interpolated_y[::-1]]))
    qVals_FDR = p_adjust_by(tensorBatch)

    res_index = calc_res_index(qVals_FDR, p_cutoff, False)[0] # p 0.05 for median resolution
    if res_index < 0:
        actualRes_global = lowResRounded
    else:
        actualRes_global = int((sampled_x[::-1][res_index]) * 100) / 100  # Round to 2 decimal places
    return 1-torch.mean(qVals_FDR), actualRes_global

def write_medianRes(
    resPerZSlice_dict: dict[float, list[float]],
    ratioSignal: float,
    resolutions: list[float],
    p_cutoff: float,
    lowRes: float,
    config: str,
    mode: str,
    outputDir: str,
    preName: str,
    mask_measure: str,
) -> float:
    """Compute the median resolution, produce diagnostic plots, and write results.

    Averages the per-slice p-values for each resolution
    shell, passes the resulting curve through FDR-corrected resolution
    fitting, generates the corresponding q-value plot (and, for
    tomograms or tilt-series, also a p-value heatmap), serialises the
    raw per-slice dictionary to JSON, and returns the estimated
    resolution.

    Parameters
    ----------
    resPerZSlice_dict : dict of {float : list of float}
        Mapping from resolution to a list of per-slice summary
        statistics (median or mean p-values), as produced by
        :func:`calculate_median_res`.
    ratioSignal : float
        Fraction of voxels inside the signal mask, displayed in the
        plot titles.
    resolutions : list of float
        Resolution value for each shell (same order as
        *resPerZSlice_dict* keys).
    p_cutoff : float
        Significance threshold passed to :func:`getFittedResolution`
        for the FDR correction.
    lowRes : float
        Low-resolution fallback limit used when no shell passes the
        significance threshold.
    config : str
        Acquisition type – ``"Tomograms"``, ``"Tilt-Series"``,
        ``"Micrographs"``, or ``"Refined-maps"``.  A p-value heatmap
        is only generated for tomograms and tilt-series.
    mode : str
        Run mode (e.g. ``"batch"``).  Console output is suppressed
        when running in batch mode.
    outputDir : str
        Directory where plots and the JSON file are written.
    preName : str
        Filename prefix for all output files.
    mask_measure : str
        Label for the summary statistic (``"median"`` or
        ``"average"``), printed to the console.

    Returns
    -------
    actualRes_global_new : float
        Estimated median resolution (truncated to two decimal places).
    """
    
    pValListGlobal = []
    lowResRounded = int(lowRes*100)/100
    for i in resPerZSlice_dict:
        median_p_per_res = np.mean(resPerZSlice_dict[i])
        pValListGlobal.append(median_p_per_res)

    pVals_qual, actualRes_global_new = getFittedResolution(resolutions, pValListGlobal, lowResRounded, p_cutoff)
    if (config == "Tomograms") or (config == "Tilt-Series"): 
        plot_heatmap_pvalue(resPerZSlice_dict, os.path.join(outputDir, preName + "_pValuePlot"), 0, 0.05, "Slices", "Resolution", "p-Value", 7, 4, "svg", actualRes_global_new, ratioSignal)

    plot_heatmap_pvalue_median(resolutions, pValListGlobal, os.path.join(outputDir, preName + "_medianPValuePlot"), 0, 0.5, "1/Resolution", mask_measure, 8, 5, "svg", actualRes_global_new, ratioSignal)
    if mode != "batch": print(str(mask_measure) + " resolution calculated in signal regions: " + str(actualRes_global_new))
    if mode != "batch": print("ratio of considered signal regions: " + str(ratioSignal))  
    with open(os.path.join(outputDir, preName + "_rawPValues.json"), "w") as json_file:
        json.dump(resPerZSlice_dict, json_file, indent=4)

    return actualRes_global_new