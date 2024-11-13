# Multi-scale decomposition of sea surface height snapshots using machine learning

This repository contains the implementation of machine learning models for decomposing Sea Surface Height (SSH) snapshots into balanced motions (BMs) and unbalanced motions (UBMs), as described in our paper ["Multi-scale decomposition of sea surface height snapshots using machine learning"](https://arxiv.org/abs/2409.17354).

## Overview

Ocean circulation plays a crucial role in weather prediction, climate understanding, and blue economy management. Our work focuses on decomposing Sea Surface Height (SSH) observations into balanced and unbalanced motions, which is particularly relevant for data from the SWOT satellite's high-resolution measurements.

### Key Features
- Implementation of deep learning models for SSH decomposition
- Zero-phase component analysis (ZCA) whitening
- Data augmentation techniques for improved multi-scale fidelity
- Methods for handling limited training data in oceanographic applications

## Authors
- Jingwen Lyu
- Yue Wang
- Christian Pedersen
- Spencer Jones
- Dhruv Balwada

Code implementation by Jingwen Lyu and Yue Wang.


## Data
Provided by Spencer Jones https://doi.org/10.1029/2022MS003220
Data link: https://zenodo.org/records/7495109

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{lyu2024multi,
  title={Multi-scale decomposition of sea surface height snapshots using machine learning},
  author={Lyu, Jingwen and Wang, Yue and Pedersen, Christian and Jones, Spencer and Balwada, Dhruv},
  journal={arXiv preprint arXiv:2409.17354},
  year={2024}
}
```
