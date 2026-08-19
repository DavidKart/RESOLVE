# RESOLVE
Tool for resolution estimation in cryo-EM tomograms, tilt-series, micrographs and STA/SPA maps.

## Usage
Instructions and tutorial: [Usage](Instructions.md)

## Installation for Linux

### Downloading/Cloning
```bash
git clone https://github.com/DavidKart/RESOLVE.git
cd RESOLVE
```

### Creating python environment

```bash
conda create -n resolve python=3.11
conda activate resolve
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
pip install .
```

### Verifying CUDA installation
Check if CUDA is properly configured:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('PyTorch version:', torch.__version__)"
```

If CUDA is available, you should see your GPU(s) listed. If not, please check your CUDA driver installation.
