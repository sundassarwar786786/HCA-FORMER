# HCA-Former: A Context-Aware Transformer for Adaptive Thermal Comfort Prediction

## This is a pytorch implementation of the following paper:
Sundas Sarwar, Jerry Chun-Wei Lin, Lars Arne Jordanger, Daniela Annunziata, Francesco Piccialli

## 🔍 Abstract

Ensuring occupant well-being, improving energy efficiency, and optimizing indoor environments based on accurate thermal comfort predictions constitute key objectives in smart buildings. However, traditional models such as the Predicted Mean Value (PMV) and adaptive approaches often rely on static assumptions and fail to capture the complex interplay between environmental conditions, occupant preferences, and contextual factors such as climate, season, and building characteristics. We present a Hierarchical Context Attention (HCA)-based approach that leverages multisource contextual data, including building attributes, geographic information, environmental parameters, and seasonal fluctuations, together with transformer-based deep learning to address these limitations. The framework employs HCA to dynamically model dependencies across contextual layers and utilizes Multi-Head Self-Attention (MHSA) to capture intricate interrelationships among thermal comfort features while eliminating redundancy. The experimental results demonstrate that our model is more efficient than state-of-the-art approaches, achieving a high accuracy of 0.802 and an F1-score of 0.795 for multiclass thermal comfort prediction. In addition, the ROC analysis and confusion matrix results further validated its robustness and generalization across diverse indoor conditions. For adaptive Heating, Ventilation and Air Conditioning (HVAC) optimization and next-generation smart building systems, the proposed HCA framework provides context-aware, data-efficient, and scalable solutions.



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





