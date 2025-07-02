# CiDer 🍹

This repository implements **CiDer: A Black-box Approach to Classify Node with Certified Robustness Guarantees** in **IEEE INFOCOM 2025**.

---

## 🧠 Paper Overview

CiDer introduces a novel **black-box certified robustness** framework for node classification. It leverages a denoiser-based model to provide provable guarantees on the classification robustness of graph nodes under adversarial perturbations. 

### 1. Illustration  
![CiDer Illustration](images/CiDer.pdf)

### 2. Pipeline Overview  
![CiDer pipeline](images/pipeline.pdf)

---

## 🛠️ Quick Start

1. Clone this repository
```bash
git clone https://github.com/xiaoyuu-liang/CiDer.git
cd CiDer
```

2. Install dependencies
```bash
pip install -r requirements.txt
```
or, use conda
```bash
conda create -n cider python=3.10
conda activate cider
pip install -r requirements.txt
```

3. Train and Test
Start from the denoiser training, then run classification and evaluation:
```bash
# Modify the configs and train the denoiser
python denoiser.py

# Train the classification model
python train_gnn.py
```

## 🤝 More Info

Feel free to contact Xiaoyu Liang for questions or collaborations.