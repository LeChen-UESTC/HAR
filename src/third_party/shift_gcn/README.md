# Shift-GCN Vendor Notes

This directory contains the minimal Shift-GCN model and NTU graph code adapted
from the official implementation:

https://github.com/kchengiva/Shift-GCN

The original project is licensed under Creative Commons
Attribution-NonCommercial 4.0 International. See `LICENSE.txt`.

Adaptation notes:

- the original CUDA temporal shift extension is replaced by a pure PyTorch
  fallback in `shift.py`;
- CUDA-only parameter initialization was changed to device-agnostic tensors;
- `forward_features()` was added to expose the pre-classifier feature map
  required by Skeleton-GIRCSE;
- deprecated NumPy integer aliases were removed.
