# HCA-Former: A Context-Aware Transformer for Adaptive Thermal Comfort Prediction

This is the official **PyTorch implementation** of the paper:

> **Sundas Sarwar, Jerry Chun-Wei Lin, Lars Arne Jordanger, Daniela Annunziata, Francesco Piccialli**  
> _HCA-Former: A Context-Aware Transformer for Adaptive Thermal Comfort Prediction_

---

## 🔍 Abstract

Ensuring occupant well-being, improving energy efficiency, and optimizing indoor environments based on accurate thermal comfort predictions constitute key objectives in smart buildings. However, traditional models such as the Predicted Mean Value (PMV) and adaptive approaches often rely on static assumptions and fail to capture the complex interplay between environmental conditions, occupant preferences, and contextual factors such as climate, season, and building characteristics.

We present **HCA-Former**, a transformer-based framework that introduces **Hierarchical Context Attention (HCA)** and **Multi-Head Self-Attention (MHSA)** to dynamically model dependencies across contextual layers. The proposed model eliminates redundancy and enhances generalization by capturing intricate interrelationships among thermal comfort features.

### 🔬 Key Contributions

- ✅ **Multiclass prediction** of thermal comfort levels.
- 🧠 **Context-aware deep learning** with transformer architecture.
- 🏢 **Scalable** across building types, climates, and seasons.
- 📊 Achieves **0.802 accuracy** and **0.795 F1-score** on real-world data.
- 🌀 Robust performance verified via ROC and confusion matrix analysis.
- 🌐 Ideal for **smart HVAC systems** and real-time thermal comfort control.

---

## 🛠️ Setup & Installation

Install dependencies using either **Conda**.

### Using Conda

```bash
# 1. Create and activate a conda environment
conda create -n hcaformer_env python=3.8 -y
conda activate hcaformer_env

# 2. Install PyTorch.
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch


---
## Requirements

python>=3.6
pytorch>=0.4

## Note
Follow the tutorial here to download the ASHRAE Global Thermal Comfort Database II and Scales Project, place the files into the "dataset" zip folder.





