# HCA-Former: A Context-Aware Transformer for Adaptive Thermal Comfort Prediction

This is the official **PyTorch implementation** of the paper:

> **Sundas Sarwar, Jerry Chun-Wei Lin, Lars Arne Jordanger, Stefano Izzo, Daniela Annunziata, Francesco Piccialli**  

---

## Abstract

Ensuring occupant well-being, improving energy efficiency, and optimizing indoor environments through accurate thermal comfort prediction are critical challenges in smart building systems. However, traditional models such as the Predicted Mean Vote (PMV) and adaptive approaches often rely on static assumptions and fail to capture complex interactions between environmental conditions, occupant preferences, and contextual factors such as climate, season, and building characteristics.

We propose **HCA-Former**, a context-aware transformer-based framework that introduces a **Hierarchical Context Attention (HCA)** mechanism to model structured dependencies among contextual variables through learnable embeddings and sequential gated fusion. The framework integrates HCA-based cross-attention with convolution-enhanced **Multi-Head Self-Attention (MHSA)** to jointly model hierarchical contextual relationships and environmental feature interactions.

Experimental results on the **ASHRAE Global Thermal Comfort Database II** and the **Scales** dataset, evaluated across five experimental runs, show that HCA-Former outperforms the evaluated machine learning (ML), transformer-based, and recent state-of-the-art baselines:

| Dataset | Accuracy | F1-score | Gain over strongest baseline | Welch's *t*-test |
|---|---|---|---|---|
| ASHRAE | 0.804 | 0.799 | +3.0 pp | *p* = 1.1 × 10⁻⁸ |
| Scales | 0.807 | 0.805 | +5.2 pp | *p* = 2.3 × 10⁻⁵ |

Ablation experiments show the complementary contributions of hierarchical contextual modeling, MHSA, and contextual and sensor-derived features, with the largest performance degradation observed when contextual features are excluded. These findings highlight the effectiveness of structured context-aware representation learning for adaptive thermal comfort prediction and its potential application in intelligent HVAC systems.

### Key Contributions

- **Hierarchical Context Attention (HCA):** contextual variables (country, climate, season, city, building type) are encoded with learnable embeddings and fused sequentially through learnable sigmoid gates.
- **Context-aware sensor fusion:** HCA cross-attention lets environmental features (queries) attend to the hierarchical context representations (keys and values).
- **Convolution-enhanced MHSA:** Conv1D-based Q/K/V projections capture feature-level dependencies among context-aware representations.
- **Multiclass prediction** of thermal sensation on a 5-point scale (−2 to +2).
- **Evaluated on two public datasets** (ASHRAE Global Thermal Comfort Database II and Scales) over five experimental runs, with statistical significance testing and ablation studies.
- Suitable for **smart HVAC systems** and real-time thermal comfort control.


---

## Setup & Installation

Install dependencies using **Conda**.


# 1. Create and activate a conda environment
```bash
conda env create -f environment.yml
```

or follow these instructions

```bash
conda create -n hcaformer_env python=3.8 -y
conda activate hcaformer_env
conda install -c conda-forge pytorch pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

## Requirements
Before running the project, ensure have the following dependencies installed:
- Python
- PyTorch
- CUDA Toolkit (for GPU support)
- imbalanced-learn
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn


## Execution Steps

This project consists of two primary steps: **data preprocessing** and **model training with cross-validation**.


### Step 1: Preprocess the Data

Use `ASHRAE_preprocessing.py` to clean and prepare the ASHRAE dataset for training.
```bash
python -u ASHRAE_preprocessing.py
```

Instead, use `Scales_Dataset_preprocessing.py` to clean and prepare the Scales_Dataset for training
```bash
python -u Scales_Dataset_preprocessing.py
```

### Step 2: Train and Evaluate the Model
Run `ASHRAE_train.py` to train and evaluate the model on ASHRAE dataset using k-fold cross-validation.
```bash
python -u ASHRAE_train.py
```

Instead, run `Scales_Dataset_train.py` to train and evaluate the model on Scales_Dataset dataset using k-fold cross-validation.
```bash
python -u Scales_Dataset_train.py
```

## Note

This project requires the following datasets:

1. **ASHRAE Global Thermal Comfort Database II**
2. **Scales Project Dataset**

**Instructions**:

- Both datasets are provided in a zip archive.
- Unzip the archive and **place the extracted contents into the** `./data/` **folder** if the extraction does not do so by default.

