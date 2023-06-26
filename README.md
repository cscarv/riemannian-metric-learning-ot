# Riemannian Metric Learning via Optimal Transport

This repository is the official implementation of [Riemannian Metric Learning via Optimal Transport](https://arxiv.org/abs/2205.09244), which appeared as a poster at ICLR 2023.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

The training data for the metric learning experiments and bird migration experiments are in the ```data``` folder. No additional steps need to be taken to pre-process this data.

The training data for the scRNA experiments is hosted on Google Drive at ```https://drive.google.com/file/d/1VC9i5gvZAxCE-RkydXHdanXohY6OGO5P/view?usp=sharing```. The first cell in ```scrna_experiments.ipynb``` uses ```gdown``` to download this dataset to the correct folder.

## Training and Evaluation

To repeat the metric learning experiments in Section 5.1 of the paper, run the cells in ```metric_recovery_experiments.ipynb``` in their given order.

To repeat the scRNA experiments in Section 5.2 of the paper, run the cells in ```scrna_experiments.ipynb``` in their given order.

To repeat the bird migration experiments in Section 5.3 of the paper, run the cells in ```snow_goose_experiments.ipynb``` in their given order.

## Pre-trained Models

Pre-trained models are included in the ```trained_models``` folder. In particular, the parameters of the metric tensor used for the scRNA experiments in Section 5.2 are stored in ```trained_models/scrna_pretrained_params.pt```. We have also included pre-trained velocity fields for scRNA trajectory inference (learned with and without the metric tensor) in ```trained_models/scrna_vel_fields```.
