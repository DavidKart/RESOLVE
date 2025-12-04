# RESOLVE
Tool for resolution estimation in cryo-EM tomograms, micrographs and STA/SPA maps.

## Usage
Instructions and tutorial: [Usage](Instructions.md)

## Installation for Linux

### Downloading/Cloning
```bash
git clone https://github.com/Dakat/RESOLVE.git
cd RESOLVE
```

### Creating python environment

#### Option 1: Use the provided environment file (recommended)
```bash
conda env create -f environment_minimal.yml
conda activate RESOLVE_env
```

#### Option 2: Manual installation
Create a new conda environment and install all required packages:
```bash
# Create and activate environment
conda create -n RESOLVE_env python=3.10
conda activate RESOLVE_env

# Install conda packages
conda install numba numba-cuda cuda-toolkit

# Install pip packages
pip install matplotlib mrcfile numpy pyfftw pyqt5 scikit-image scipy seaborn tifffile
```

### Verifying CUDA installation
Check if CUDA is properly configured:
```bash
python -c "from numba import cuda; print('CUDA available:', cuda.is_available()); print('GPUs:', [gpu.name for gpu in cuda.gpus] if cuda.is_available() else 'None')"
```

If CUDA is available, you should see your GPU(s) listed. If not, please check your CUDA driver installation.
