# Diffusion Models for Time Series Imputation and Forecasting

## Introduction
This folder is part of a series of implementations focusing on diffusion models applied to time series data, specifically using the CSDI (Conditional Score-based Diffusion Model) framework. These implementations are designed to provide insights and practical experience with state-of-the-art models in time series data forecasting and imputation.

### Notebooks
- **csdi_forecasting.ipynb** - Demonstrates the use of the CSDI model for time series forecasting tasks. It includes a brief description of the algorithm, its implementation, and a demonstration of how to use it for forecasting.
- **csdi_imputation.ipynb** - Focuses on the application of the CSDI model for data imputation in time series. It includes a brief description of the algorithm, its implementation, and a demonstration of how to use it for imputation.

### Code
This section includes code structure and description of the files:

- [config/](https://github.com/VectorInstitute/diffusion_model_bootcamp/tree/gs_bootcamp/reference_implementations/time_series_reference_impelementation/CSDI/config) ==>  Contains configuration files that specify model parameters and experimental settings.
- [dataset/](https://github.com/VectorInstitute/diffusion_model_bootcamp/tree/gs_bootcamp/reference_implementations/time_series_reference_impelementation/CSDI/dataset)
  -  [dataset_download.py](https://github.com/VectorInstitute/diffusion_model_bootcamp/blob/main/reference_implementations/time_series_reference_impelementation/CSDI/dataset/dataset_download.py) ==>  For downloading the dataset.
  - [dataset_forecasting.py](https://github.com/VectorInstitute/diffusion_model_bootcamp/blob/gs_bootcamp/reference_implementations/time_series_reference_impelementation/CSDI/dataset/dataset_forecasting.py) ==>  Handles loading and preprocessing data for forecasting tasks.
  - [dataset_physio.py](https://github.com/VectorInstitute/diffusion_model_bootcamp/blob/gs_bootcamp/reference_implementations/time_series_reference_impelementation/CSDI/dataset/dataset_physio.py) ==>  Manages physiological data specific to time series imputation analysis.
  - [dataset_pm25.py](https://github.com/VectorInstitute/diffusion_model_bootcamp/blob/gs_bootcamp/reference_implementations/time_series_reference_impelementation/CSDI/dataset/dataset_pm25.py) ==>  Dedicated to handling PM2.5 environmental data sets for imputation.
- [downloads/](https://github.com/VectorInstitute/diffusion_model_bootcamp/tree/gs_bootcamp/reference_implementations/time_series_reference_impelementation/CSDI/downloads)
  - [download.py](https://github.com/VectorInstitute/diffusion_model_bootcamp/blob/gs_bootcamp/reference_implementations/time_series_reference_impelementation/CSDI/downloads/download.py) ==>  Script used for downloading necessary data files for the experiments.
- [experiments/](https://github.com/VectorInstitute/diffusion_model_bootcamp/tree/gs_bootcamp/reference_implementations/time_series_reference_impelementation/CSDI/experiments)
  - [exe_forecasting.py](https://github.com/VectorInstitute/diffusion_model_bootcamp/blob/gs_bootcamp/reference_implementations/time_series_reference_impelementation/CSDI/experiments/exe_forecasting.py) ==>  Script for running forecasting experiments using the CSDI model.
  - [exe_physio.py](https://github.com/VectorInstitute/diffusion_model_bootcamp/blob/gs_bootcamp/reference_implementations/time_series_reference_impelementation/CSDI/experiments/exe_physio.py) ==>  Script for executing experiments on physiological data for time series imputation.
  - [exe_pm25.py](https://github.com/VectorInstitute/diffusion_model_bootcamp/blob/gs_bootcamp/reference_implementations/time_series_reference_impelementation/CSDI/experiments/exe_pm25.py) ==>  Script for running experiments with PM2.5 data for time series imputation.
- [figures/](https://github.com/VectorInstitute/diffusion_model_bootcamp/tree/gs_bootcamp/reference_implementations/time_series_reference_impelementation/CSDI/figures) ==>  Contains the figures to be used in the notebooks.
- [save/](https://github.com/VectorInstitute/diffusion_model_bootcamp/tree/gs_bootcamp/reference_implementations/time_series_reference_impelementation/CSDI/save) ==>  Directory where model weights and training checkpoints are saved.
- [util/](https://github.com/VectorInstitute/diffusion_model_bootcamp/tree/gs_bootcamp/reference_implementations/time_series_reference_impelementation/CSDI/util) ==>  Utility scripts that include helper functions and additional utilities needed for data manipulation and model setup.

### Source Code
- [diff_models.py](https://github.com/VectorInstitute/diffusion_model_bootcamp/blob/gs_bootcamp/reference_implementations/time_series_reference_impelementation/CSDI/diff_models.py) ==>  Contains the implementation of diffusion model specific to the algorithm.
- [main_model.py](https://github.com/VectorInstitute/diffusion_model_bootcamp/blob/gs_bootcamp/reference_implementations/time_series_reference_impelementation/CSDI/main_model.py) ==>  The main execution script that sets up, trains, and evaluates the diffusion models based on the configuration provided.

## Getting Started
To start working with the materials and code in this topic:
1. Ensure you have followed the reference to the installation guide and environment setup in \docs.
2. Move to notebook `csdi_imputation.ipynb` to learn about CSDI algorithm and its implementation to imputation. Please first make sure to set the kernel to `diffusion_model` in the notebook. Further run the code to download data.
3. Move to notebook `csdi_forecasting.ipynb` to learn about the implementation of CSDI to forecasting task. Please first make sure to set the kernel to `diffusion_model` in the notebook.
