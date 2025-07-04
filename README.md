# HCA-Former: A Context-Aware Transformer for Adaptive Thermal Comfort Prediction

This is the official **PyTorch implementation** of the paper:

> **Sundas Sarwar, Jerry Chun-Wei Lin, Lars Arne Jordanger, Daniela Annunziata, Francesco Piccialli**  

---

## Abstract

Ensuring occupant well-being, improving energy efficiency, and optimizing indoor environments based on accurate thermal comfort predictions constitute key objectives in smart buildings. However, traditional models such as the Predicted Mean Value (PMV) and adaptive approaches often rely on static assumptions and fail to capture the complex interplay between environmental conditions, occupant preferences, and contextual factors such as climate, season, and building characteristics.

We present **HCA-Former**, a transformer-based framework that introduces **Hierarchical Context Attention (HCA)** and **Multi-Head Self-Attention (MHSA)** to dynamically model dependencies across contextual layers. The proposed model eliminates redundancy and enhances generalization by capturing intricate interrelationships among thermal comfort features.

###  Key Contributions

- ✅ **Multiclass prediction** of thermal comfort levels.
- 🧠 **Context-aware deep learning** with transformer architecture.
- 🏢 **Scalable** across building types, climates, and seasons.
- 🌀 Robust performance verified via ROC and confusion matrix analysis.
- 🌐 Ideal for **smart HVAC systems** and real-time thermal comfort control.


---

## Setup & Installation

Install dependencies using **Conda**.


# 1. Create and activate a conda environment
conda create -n hcaformer_env python=3.8 -y
conda activate hcaformer_env

# 2. Install PyTorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
 Requirements

## Requirements

Before running the project, ensure have the following dependencies installed:

- Python >= **3.6**
- PyTorch >= **0.4**
- CUDA Toolkit (for GPU support): **cudatoolkit=11.3**


## Execution Steps

This project consists of two primary steps: **data preprocessing** and **model training with cross-validation**.


### Step 1: Preprocess the Data

Use `preprocessing.py` to clean and prepare the dataset for training.

python preprocessing.py --input ./data/ashrae_db2.01.csv --output ./data/processed.csv

### Step 2: Train and Evaluate the Model

Run `hca-former.py` to train and evaluate the model using k-fold cross-validation.

python hca-former.py --data ./data/processed.csv --folds 10 --epochs 100




## Note


This project requires the following datasets:

1. **ASHRAE Global Thermal Comfort Database II**
2. **Scales Project Dataset**

**Instructions**:

- Both datasets are provided in a zip archive.
- Unzip the archive and **place the extracted contents into the** `./data/dataset/` **folder**.
  





