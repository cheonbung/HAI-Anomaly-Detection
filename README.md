# 📘 Advancing Autoencoder Architectures for Enhanced Anomaly Detection in Multivariate Industrial Time Series

[![Language](https://img.shields.io/badge/language-English-orange.svg)](./README.md)
[![Language](https://img.shields.io/badge/language-Korean-blue.svg)](./README_KR.md)

This repository contains the official implementation and experimental code for the paper **"Advancing Autoencoder Architectures for Enhanced Anomaly Detection in Multivariate Industrial Time Series,"** published in **Computers, Materials & Continua (CMC) 2024**.

We propose **ConvBiLSTM-AE**, a hybrid autoencoder model for Industrial Control Systems (ICS), utilizing the **HAI (HIL-based Augmented ICS Security) 23.05** dataset. Furthermore, we provide a comprehensive benchmarking environment comparing our model against various state-of-the-art (SOTA) anomaly detection models.

---

## 1. Key Contributions

*   **ConvBiLSTM-AE Proposal**: A hybrid autoencoder model combining the spatial feature extraction capabilities of **CNN** with the temporal context learning of **BiLSTM**.
*   **Comprehensive Benchmark**: Performance comparison against various baselines (Linear, CAE, LSTM, BiLSTM) and SOTA models using the HAI 23.05 dataset.
*   **Advanced Analysis**:
    *   **Preprocessing**: Automated removal of multicollinear variables based on **VIF (Variance Inflation Factor)**.
    *   **Latent Space Visualization**: Analysis of embedding spaces using **UMAP** and **PCA**.
    *   **Reconstruction Error Analysis**: Robust detection using **Moving Average** filtering.

---

## 2. Project Structure

```text
HAI-Anomaly-Detection/
│
├── configs/
│   └── config.json                # Model hyperparameters and data paths
│
├── data/                          # Dataset directory
│   ├── hai-23.05/                 # Raw CSV files (train1~4, test1~2)
│   └── outputs/                   # Training results, weights, logs
│
├── models/                        # [Proposed & Baselines] (TF/Keras)
│   ├── __init__.py
│   ├── layers.py                  # Custom layers (e.g., Attention)
│   └── architectures.py           # ConvBiLSTM-AE, BiGRU-AE, etc.
│
├── comparisons/                   # [SOTA Benchmarks] (PyTorch)
│   ├── models/
│   │   ├── mtad_gat/              # Graph Attention Network
│   │   ├── omni_anomaly/          # Stochastic RNN (VAE)
│   │   ├── tran_ad/               # Transformer + Adversarial
│   │   ├── usad/                  # Unsupervised Adversarial AE
│   │   ├── daemon/                # Adversarial AE (Double Discriminator)
│   │   └── madgan/                # LSTM-GAN
│   │
│   ├── train_mtad_gat.py          # Execution scripts for SOTA models
│   ├── train_omni_anomaly.py
│   ├── train_tran_ad.py
│   ├── train_usad.py
│   ├── train_daemon.py
│   └── train_madgan.py
│
├── utils/                         # Common Utilities
│   ├── __init__.py
│   ├── preprocessing.py           # Data loading, normalization, VIF, windowing
│   ├── metrics.py                 # F1-Score, Threshold optimization, eTaPR
│   └── visualization.py           # Loss plots, ROC/PR curves, PCA/UMAP
│
├── train.py                       # [Main] Train Proposed Model
├── evaluate.py                    # [Main] Evaluate Proposed Model
├── analysis_eda.py                # [Analysis] EDA & Embedding Analysis
├── requirements.txt               # Dependencies
└── README.md                      # Documentation (English)
```

---

## 3. Setup

### 3.1. Prerequisites
*   Python 3.8+
*   **TensorFlow 2.x** (For Main Models)
*   **PyTorch 1.8+** (For Comparison Models)
*   NVIDIA GPU (CUDA recommended)

### 3.2. Installation
```bash
git clone <repository_url>
cd HAI-Anomaly-Detection
pip install -r requirements.txt
```

### 3.3. Dataset Preparation
*   Download the **HAI 23.05** dataset and place it in the `data/hai-23.05/` directory.
    *   Github: [https://github.com/icsdataset/hai](https://github.com/icsdataset/hai)

---

## 4. Proposed Model Experiments

Train and evaluate the proposed **ConvBiLSTM-AE** and other baseline models (Linear, CAE, LSTM, BiLSTM).

### 4.1. Training (`train.py`)
```bash
# Train ConvBiLSTM-AE (Default settings)
python train.py --model Conv_BiLSTM_AE --epochs 60

# Example: Train another baseline model
python train.py --model BiLSTM_AE
```

### 4.2. Evaluation (`evaluate.py`)
Load the trained model and measure anomaly detection performance (F1, AUC, Precision, Recall) on Test 1 and Test 2 datasets.
```bash
python evaluate.py
```

### 4.3. Analysis (`analysis_eda.py`)
Perform correlation analysis (Heatmap) and **PCA visualization** of the latent vectors extracted from the trained encoder.
```bash
python analysis_eda.py
```

---

## 5. Comparative Experiments (Benchmarks)

We provide PyTorch-based implementations for performance comparison with the latest SOTA models.

| Model | Feature | Command |
| :--- | :--- | :--- |
| **MTAD-GAT** | Spatio-temporal correlation modeling via Graph Attention Network | `python comparisons/train_mtad_gat.py` |
| **OmniAnomaly** | Stochastic modeling based on GRU-VAE | `python comparisons/train_omni_anomaly.py` |
| **TranAD** | Transformer + Adversarial Training | `python comparisons/train_tran_ad.py` |
| **USAD** | AutoEncoder + GAN (Adversarial Training) | `python comparisons/train_usad.py` |
| **DAEMON** | Adversarial AE (Reconstruction & Latent Discriminator) | `python comparisons/train_daemon.py` |
| **MAD-GAN** | LSTM-GAN (Includes Latent Space Optimization) | `python comparisons/train_madgan.py` |

> **Note**: All results from comparative experiments are saved in the `data/outputs/[model_name]/` directory.

---

## 6. Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{lee2024advancing,
  title={Advancing Autoencoder Architectures for Enhanced Anomaly Detection in Multivariate Industrial Time Series},
  author={Lee, Byeongcheon and Kim, Sangmin and Maqsood, Muazzam and Moon, Jihoon and Rho, Seungmin},
  journal={Computers, Materials & Continua},
  volume={81},
  number={1},
  pages={1275--1302},
  year={2024},
  publisher={Tech Science Press},
  doi={10.32604/cmc.2024.054826}
}
```

---

## 7. Patent

The results of this research have been filed with the Korean Intellectual Property Office (KIPO).

*   **Title**: METHOD FOR ANOMALY DETECTING BASED ON DEEP LEARNING MODEL IN TIME SERIES DATA RELATED TO MULTIVARIATE INDUSTRIAL THINGS TERMINALS, AND APPARATUS THEREOF
*   **Application Number**: 10-2024-0161756
*   **Date of Application**: 2024.11.14
*   **Applicant**: Industry-University Cooperation Foundation, Chung-Ang University
*   **Inventors**: Sangmin Kim, Byeongcheon Lee, Jihoon Moon, Seungmin Rho, Muazzam Maqsood

---

## 8. License

This work is licensed under a **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

*   **Copyright**: Byeongcheon Lee, Sangmin Kim, Muazzam Maqsood, Jihoon Moon, Seungmin Rho
*   **Source**: Computers, Materials & Continua (CMC), 2024, vol.81, no.1.

*Note: The code for SOTA comparison models in the `comparisons/` folder follows the licensing policies of their respective original authors. Please refer to the original repositories for details.*