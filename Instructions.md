# Instructions for RESOLVE usage and tutorial

## Installation

See [README](README.md)

## Table of Contents

### 1. [Usage](#usage)
- [1.1 GUI mode](#gui-mode)
- [1.2 Command line mode](#command-line-mode)

### 2. [Tutorial on test dataset](#tutorial-on-test-dataset)
- [2.1 SPA map with command-line processing](#spa-map-with-command-line-processing)
- [2.2 Micrograph dataset with command-line batch processing](#micrograph-dataset-with-command-line-batch-processing)
- [2.3 Notes about tilt-series and tomogram batch processing](#notes-about-tilt-series-and-tomogram-batch-processing)

---

## Usage

GUI mode is convenient for processing a single pair of half-maps. Command line mode also offers a batch processing mode, allowing RESOLVE to run on a complete dataset and producing a summary output file.

### GUI mode

Activate conda environment:
```bash
conda activate resolve
```

Open graphical user interface (GUI) from anywhere:
```bash
RESOLVE_GUI
```

The GUI should appear now:

![GUI](screenshots/GUI.png)

On the top, the drop-down menu offers 4 processing options:
- `Refined-Maps` (referring to SPA/STA maps)
- `Micrographs`
- `Tilt-series`
- `Tomograms`

Please make sure to always choose the correct option before you proceed - wrong input types are not necessarily caught and will lead to incorrect results!

**`Input 1`** and **`Input 2`** refer to the input half-map pair.

**`Pixel Size`** should be read from the header - if this is not correct, please input the correct pixel size here!

**`Output Directory`** Choose where to save the results.

There are a few advanced processing options (opened when clicking `Show Advanced Options`):

![GUI advanced processing options](screenshots/GUI_advanced.png)

**`CPU Threads`** Specify number of CPU threads. For CPU batch processing, increased number of threads will significantly increase processing speed.

**`GPU`** Choose whether to enable GPU processing or not, and option to choose GPUs. For micrographs, GPU mode is not recommended. Otherwise, GPU usage will make processing usually much faster. Probably 2 GPUs maximize efficiency for most cases.

**`Run fast`** Possibly less accurate, but faster - lower sampling in Fourier space (fewer resolutions checked) and real space (increased step size). Requires also less memory.

#### Settings for median global resolution estimates
Median resolution is intended as a global quality for Micrographs, tilt-series or tomograms.

**`Masking strategy`** Choose how to mask the map for global resolution estimation. Options: 'remove_background' (Default. Automatically remove regions not passing the lowest measured resolution shell. This may be useful for partially empty input maps.), 'signal_mask' (provide a custom binary mask file to focus the measure on a region of interest), 'full_map' (use the entire map without masking).

**`Input Mask`** Optional input when using 'signal mask' as a masking strategy. The input mask need to have the same dimensions as the input half-maps.

**`Measure`** The mean may be chosen instead of the median.

### Command line mode

Activate conda environment:
```bash
conda activate resolve
```

Go into help mode to find all available processing options:
```bash
RESOLVE --help
```

All options previously described for the GUI are also available for command-line processing. Additionally, there is a batch processing mode.

To activate the batch processing mode, use the argument `--mode batch` (instead of `--mode single`)

RESOLVE now does not require the two half-maps as input (`--input1` and `--input2`). Instead, please provide an input directory (`--inputDir`) where all the half-maps of your dataset are located. Then, provide a unique identifier via `--odd_id` and `--even_id`, for example `ODD` and `EVN` or `half1` and `half2`. RESOLVE expects both half-maps to be named identically apart from this identifier.

---

## Tutorial on test dataset

### SPA map with command-line processing

Navigate to tutorial_data:
```bash
cd tutorial_data
```

Run RESOLVE from command line:
```bash
RESOLVE --mode single --config "Refined-Maps" --input1 STA/emd_34658_half_map_1.map --input2 STA/emd_34658_half_map_2.map --outputDir STA_output --gpu_enabled
```

The output can be found in the newly created STA_output directory. Note that both `map` as well as `mrc` endings should work.

### Micrograph dataset with command-line batch processing

Navigate to tutorial_data:
```bash
cd tutorial_data
```

Run RESOLVE from command line:
```bash
RESOLVE --mode batch --config "Micrographs" --inputDir micrographs --odd_id ODD --even_id EVN --outputDir micrograph_batch_output --cpu_threads 16
```

For faster processing (skipping some shells and increased step size), add the `--fast` option:
```bash
RESOLVE --mode batch --config "Micrographs" --inputDir micrographs --odd_id ODD --even_id EVN --outputDir micrograph_batch_output --cpu_threads 16 --fast
```

For GPU batch processing, instances will be split across GPUs.

The output can be found in the newly created micrograph_batch_output directory. For each input micrograph half-map pair, there will be an output .mrc file, an output .png file and a *q*-value plot (with the median *q*-value per resolution). Note that the cutoff for median (global) resolution determination is at *q* = 0.05. Note also the summary.tsv file, the summarized and sorted output for the median (global) resolution for all input micrographs.

### Notes about tilt-series and tomogram batch processing

For tilt-series and tomograms, please adjust the `--config` option accordingly to either `--config Tilt-series` or `--config Tomograms`. For 3D input, always add the `--gpu_enabled` flag, otherwise processing may take very long. Consider using the `--fast` flag if you have a large dataset to further speed up processing.
