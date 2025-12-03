# RESOLVE
Tool for resolution estimation in cryo-EM tomograms, micrographs and STA/SPA maps

# Usage
Instructions and tutorial [Usage](Instructions.md)


# Installation for Linux

## Cloning
```https://github.com/DavidKart/RESOLVE.git
cd RESOLVE```

## Creating python Environment

### Use the provided yaml
```conda env create -f environment_minimal.yml```


### Or install libraries manually:
  - python=3.10
  - numba=0.62
  - numba-cuda=0.22
  - cuda-toolkit=12.6
  - pip:
      - matplotlib
      - mrcfile
      - networkx
      - numpy
      - pandas
      - pillow
      - pyfftw
      - pyqt5
      - scikit-image
      - scipy
      - seaborn
      - tifffile

## Verifying numba cuda
```python -c "from numba import cuda; print('CUDA available:', cuda.is_available()); print('GPUs:', [gpu.name for gpu in cuda.gpus] if cuda.is_available() else 'None')"```
