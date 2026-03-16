# Cherenkov Image Denoising

This repository contains the code for testing and evaluating various deep learning denoising models on Cherenkov image data. 

## Models Implemented
This project explores several different architectures for image denoising:
* **U-Net:** Standard U-Net and U-Net with uncertainty estimation.
* **NAFNet:** Nonlinear Activation Free Network for image restoration.
* **Palette Diffusion:** Diffusion models tailored for image-to-image translation (including frequency and TAM variants).

## Repository Structure
* `Unet_*.py`: Scripts for training, inference, and uncertainty estimation using U-Net.
* `nafnet_*.py`: Model definitions and training scripts for NAFNet.
* `palette_diffusion_*.py`: Implementations of diffusion-based denoising.
* `noise_data_generation.py` & `cumulative_noise_simu_intensity.py`: Tools for simulating and generating noisy Cherenkov data.

## Note on Data
Due to size constraints and privacy, the actual dataset (`Acc_image`, `noisy_patches`, etc.) are **not** included in this repository. 

To run this code locally, you will need to generate or provide your own image datasets and place them in the appropriate root folders as referenced in the scripts.

## How to Run
*(Add your specific instructions here. For example:)*
1. Ensure change the path to your image data folder.
2. Run data generation: `python noise_data_generation.py`
3. Train a model: `python Unet_denoise.py`