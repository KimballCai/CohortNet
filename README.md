# ECHO: An Effective Cohort Discovery Framework for Interpretable Healthcare Analytics

----
This is the source code for SIGMOD 2024 submission.

Thanks for your interest about out work.

## Requirements

---
- Install python 3.6, Pytorch 1.10.2.
- If you want to use the GPU, please install the CUDA accordingly.

## Data preparation

---
As for the MIMIC3 dataset, you must submit the application for data access from [https://mimic.physionet.org/](https://mimic.physionet.org/).
After downloading the CSVs, you first need to build the benchmark dataset according to the [mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks/),
and filter features according to the [MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract).

## Train the CohortNet in the ECHO framework

---
You first need to train the CohortNet in the following two steps:
1. train the Multi-Channel Feature Learning Module for fine-grained patient representation learning with 
the jupyter file [train.ipynb](train.ipynb)
2. train the Cohort Discovery Module, Cohort Learning Module, and 
Cohort Exploitation Module for the automatic cohort discovery, comprehensive cohort learning, and personalized cohort 
exploitation with the jupyter file [train_cohort.ipynb](train_cohort.ipynb)

If you want to use your own dataset, please design the [SetLoader](dataset.py), and open the mode for the forward 
imputation (i.e. impute the missing data with the last observation) and standardization before training the CohortNet.