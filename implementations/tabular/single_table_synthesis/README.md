# Diffusion Models for Single Table Data Overview

## Introduction
This folder contains a collection of Jupyter notebooks and code samples designed to enhance your understanding of applying diffusion models for single table synthesis. For this purpose, we will explore two different algorithms: TabDDPM and TabSyn. Both algorithms are designed for single table data synthesis and have their own unique characteristics and applications. We will cover them in detail in separate notebooks and code files. We will also study the quality of generated data using various metrics for each algorithm.


## Notebooks
Here you will find the following Jupyter notebooks:
1. **tabddpm_implementation** - This notebook covers tabddpm implementation for single table synthesis. It includes a brief description of the algorithm, its implementation, and a demonstration of how to use it for sampling synthetic data.
2. **tabsyn_implementation** - This notebook covers tabsyn implementation for single table synthesis. It includes a brief description of the algorithm, its implementation, and a demonstration of how to use it for sampling synthetic data.
3. **evaluate_synthetic_data** - This notebook covers the evaluation of synthetic data generated by TabDDPM and TabSyn algorithms. In order to assess the quality of synthetic data, we report various evaluation metrics including low-dimensional statistics, high-dimensional statistics, evaluation of the synthetic data on downstream tasks and privacy metrics. This notebook includes a brief description of these evaluation metrics, their implementation, and a demonstration of how to use them to evaluate the quality of synthetic data.

## Code
This section includes code structure and description of the files:

* [data_info/](./data_info) ==> contains the json data information file that provides details about various availabe dataset and their attributes.
* [figures/](./figures) ==> contains the figures to be used in the notebooks.
* [scripts/](./scripts) ==> contains the running scripts.
    * [download_dataset.py](./scripts/download_dataset.py) ==> contains the code to download the data.
    * [process_dataset.py](./scripts/process_dataset.py) ==> contains the code to preprocess the data.
    * [main.py](./scripts/main.py) ==> contains the code to train TabDDPM and TabSyn algorithms for synthesizing and sampling.
    * [impute.py](./scripts/impute.py) ==> contains the code to trained tabsyn on imputation task.
    * [eval](./scripts/eval) ==> contains the code to evaluate the synthetic data generated by TabDDPM and TabSyn algorithms.
        * [eval_dcr.py](./scripts/eval/eval_dcr.py) ==> contains the code to report privacy metrics.
        * [eval_density.py](./scripts/eval/eval_density.py) ==> contains the code to report low-order statistics.
        * [eval_detection.py](./scripts/eval/eval_detection.py) ==> contains the code to report whether the synthetic data and real data are distinguishable.
        * [eval_impute.py](./scripts/eval/eval_impute.py) ==> contains the code to report imputation quality.
        * [eval_mle.py](./scripts/eval/eval_mle.py) ==> contains the code to report machine learning efficiency.
        * [eval_quality.py](./scripts/eval/eval_quality.py) ==> contains the code to report high-order statistics.
* [slrm](./slrm) ==> contains the code for the running the training and sampling of TabDDPM and TabSyn algorithms through slurm.
* [src](./src) ==> contains the source code.
    * [baselines.py](./src/baselines) ==> contains the code for TabDDPM and tabsyn algorithms.
        * [tabddpm.py](./src/baselines/tabddpm.py) ==> contains the code for TabDDPM algorithm. This directory includes the implementaion of architecture and training code for gaussian and multinomial diffusion models, and sampling using trained models.
        * [tabsyn.py](./src/baselines/tabsyn.py) ==> contains the code for TabSyn algorithm. This directory includes the implementaion of architecture for both variational auto-encoder and latent diffusion model, training vae, trainining diffusion model and sampling.
    * [data.py](./src/data.py) ==> contains the code for data processing and loading.
    * [metrics.py](./src/metrics.py) ==> contains the code for metrics used in training.
    * [utils.py](./src/utils.py) ==> contains the code for utility functions.


## Getting Started
To get started with the materials in this topic:
1. Ensure you have followed the reference to the installation guide and environment setup in `\docs`.
2. Move to notebook `tabddpm_implementation.ipynb` to learn about TabDDPM algorithm and its implementation. Please first make sure to set the kernel to `diffusion_model` in the notebook. Further run the code to sample synthetic data.
3. Move to notebook `tabsyn_implementation.ipynb` to learn about TabSYN algorithm and its implementation. Please first make sure to set the kernel to `diffusion_model` in the notebook. Further
4. Finally using the notebook `evaluate_synthetic_data.ipynb` to evaluate the synthetic data generated by TabDDPM and TabSyn algorithms. Please first make sure to set the kernel to `diffusion_model` in the notebook. Further run the code to evaluate the synthetic data.
