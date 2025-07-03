# HCA-Former: A Context-Aware Transformer for Adaptive Thermal Comfort Prediction

## This is a pytorch implementation of the following paper:
Sundas Sarwar, Jerry Chun-Wei Lin, Lars Arne Jordanger, Daniela Annunziata, Francesco Piccialli

## 🔍 Overview

**HCA-Former** introduces a novel **Hierarchical Context Attention (HCA)** framework that leverages multisource contextual data (such as environmental, geographic, building, and seasonal features) with a transformer-based architecture for **accurate and adaptive thermal comfort prediction** in smart building environments.

Traditional models like PMV and adaptive comfort models often assume static environments and fail to integrate dynamic contextual variables. HCA-Former overcomes these limitations through:

- A **hierarchical attention mechanism** to prioritize relevant contextual layers.
- A **Multi-Head Self-Attention (MHSA)** module to capture interdependencies among comfort-related features.
- **Robust performance** validated using ROC analysis and confusion matrix evaluation.

---

## 📈 Key Features

- ✅ Multiclass prediction of thermal comfort levels.
- 🧠 Transformer-based deep learning with contextual awareness.
- 🏢 Adaptable to different building types, climates, and seasons.
- 📊 Achieves **0.802 accuracy** and **0.795 F1-score**.
- 🌐 Suitable for real-time HVAC adaptation and smart building systems.

---
## Requirements

python>=3.6
pytorch>=0.4

## Note
Follow the tutorial here to download the ASHRAE Global Thermal Comfort Database II and Scales Project, place the files into the "dataset" zip folder.





