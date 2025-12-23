# ECG Time Series Reconstruction with VAE

This repository implements a Variational Autoencoder (VAE) for ECG time series reconstruction and generation.

## Overview

This project focuses on ECG signal reconstruction using VAE architecture, providing efficient training and inference for time series data.

## Installation

```bash
pip install -r requirements.txt
```

## Usage
### Data Processing
```bash
python data/datasets_preprocess/MIMIC/make_dataset.py
```
```bash
python -m reconstruction.data.build_dataloader
```

### Training VAE

```bash
python models/vae/vae_train_100.py --config config/MIMIC/mimic_cond.yaml --save_dir ./results/vae_100/
```

### Testing VAE

```bash
python models/vae/vae_test_100.py --config config/MIMIC/mimic_cond.yaml --save_dir ./results/vae_100/
```

### Showing reconstructed ECG
```bash
python evaluation/plot2.py
```

### Viewing evaluation metrics 
```bash
python evaluation/compute_metric.py
```

## Project Structure

```
├── models/
│   └── vae/                    # VAE model implementation
│       ├── vae_model_100.py   # VAE model definition
│       ├── vae_train_100.py   # VAE training script
│       └── vae_test_100.py    # VAE testing script
├── config/                     # Configuration files
├── data/                       # Dataset loading utilities
├── evaluation/                 # Evaluation metrics
└── utils/                      # Utility functions
```

## Main Results

Our VAE-based approach achieves efficient ECG signal reconstruction with:
- Fast training and inference
- High-quality reconstruction
- Compact latent representation

## Authors

* Code Author : [Xinyan Guan]
* Contact : [xinyan.guan.25@ucl.ac.uk]

## Citation

If you find this repo useful, please cite our work:

```bibtex
@article{guan2025reconstructing,
  title={Reconstructing 12-Lead ECG from 3-Lead ECG using Variational Autoencoder to Improve Cardiac Disease Detection of Wearable ECG Devices},
  author={Guan, Xinyan and Lai, Yongfan and Jin, Jiarui and Li, Jun and Wang, Haoyu and Zhao, Qinghao and Zhang, Deyun and Geng, Shijia and Hong, Shenda},
  journal={arXiv preprint arXiv:2510.11442},
  year={2025}
}
```

## Acknowledgement

We appreciate the following github repos for their valuable code base:

- https://github.com/pytorch/pytorch
- https://github.com/huggingface/transformers
