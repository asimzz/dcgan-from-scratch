# DCGAN from Scratch

This repository contains an implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) from scratch using PyTorch. The implementation includes both the generator and discriminator models designed to generate high-resolution images. The repository also provides training scripts and utilities to train the DCGAN on custom datasets.

## Introduction

Deep Convolutional Generative Adversarial Networks (DCGANs) are a type of GAN where both the generator and discriminator use convolutional layers. This project aims to provide a simple and clear implementation of DCGAN from scratch.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/asimzz/dcgan-from-scratch.git
cd dcgan-from-scratch
```

2. Run the `main.py` file:

```bash
python3 main.py
```

## File Descriptions

- `utils.py`: Utility functions for data loading, image saving, and other helper functions used during training and image generation.
- `models.py`: This file contains the implementation of the Generator and Discriminator classes for the DCGAN. It also includes utility functions for weight initialization and adding spectral normalization.
- `main.py`: This file contains the main script for training the DCGAN using PyTorch. It includes data loading, model initialization, training loop, and saving generated images during training.
