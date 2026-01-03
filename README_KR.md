# 📘 Advancing Autoencoder Architectures for Enhanced Anomaly Detection in Multivariate Industrial Time Series

[![Language](https://img.shields.io/badge/language-English-orange.svg)](./README.md)
[![Language](https://img.shields.io/badge/language-Korean-blue.svg)](./README_KR.md)

본 리포지토리는 **Computers, Materials & Continua (CMC) 2024**에 게재된 논문 **"Advancing Autoencoder Architectures for Enhanced Anomaly Detection in Multivariate Industrial Time Series"**의 공식 구현체 및 실험 코드를 포함함.

본 연구는 **HAI (HIL-based Augmented ICS Security) 23.05** 데이터셋을 활용하여 산업 제어 시스템(ICS)을 위한 하이브리드 오토인코더 모델인 **ConvBiLSTM-AE**를 제안함. 또한, 다양한 최신(SOTA) 이상 탐지 모델들과의 비교 실험 환경을 제공함.

---

## 1. 주요 기여 (Key Contributions)

*   **ConvBiLSTM-AE 제안**: CNN의 공간적 특징 추출 능력과 BiLSTM의 시간적 문맥 학습 능력을 결합한 하이브리드 오토인코더 모델.
*   **포괄적인 벤치마크**: HAI 23.05 데이터셋에 대해 다양한 Baseline (Linear, CAE, LSTM, BiLSTM) 및 SOTA 모델들과의 성능 비교.
*   **고도화된 분석**:
    *   **VIF (분산 팽창 요인)** 기반의 다중공선성 변수 제거 전처리 수행.
    *   **Latent Space 시각화**: UMAP 및 PCA를 이용한 임베딩 공간 분석.
    *   **Reconstruction Error 분석**: 이동 평균(Moving Average) 필터를 적용하여 탐지 강건성 확보.

---

## 2. 프로젝트 구조 (Project Structure)

```text
HAI-Anomaly-Detection/
│
├── configs/
│   └── config.json                # 모델 하이퍼파라미터 및 데이터 경로 설정
│
├── data/                          # 데이터셋 저장소
│   ├── hai-23.05/                 # HAI 23.05 원본 데이터 (train1~4, test1~2)
│   └── outputs/                   # 학습 결과, 모델 가중치, 로그 저장
│
├── models/                        # [Proposed & Baselines] (TF/Keras)
│   ├── __init__.py
│   ├── layers.py                  # Custom Attention Layer 등
│   └── architectures.py           # ConvBiLSTM-AE, BiGRU-AE 등 모델 정의
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
│   ├── train_mtad_gat.py          # 실행 스크립트
│   ├── train_omni_anomaly.py
│   ├── train_tran_ad.py
│   ├── train_usad.py
│   ├── train_daemon.py
│   └── train_madgan.py
│
├── utils/                         # 공통 유틸리티
│   ├── __init__.py
│   ├── preprocessing.py           # 데이터 로드, 정규화, VIF, Windowing
│   ├── metrics.py                 # F1-Score, Threshold 최적화, eTaPR
│   └── visualization.py           # Loss Plot, ROC/PR Curve, PCA/UMAP
│
├── train.py                       # [Main] 제안 모델(ConvBiLSTM-AE) 학습
├── evaluate.py                    # [Main] 제안 모델 평가
├── analysis_eda.py                # [Analysis] EDA 및 임베딩 분석
├── requirements.txt               # 의존성 패키지 목록
└── README_KR.md                   # 프로젝트 설명서 (국문)
```

---

## 3. 실행 환경 설정 (Setup)

### 3.1. 요구 사항
*   Python 3.8+
*   **TensorFlow 2.x** (Main Model: ConvBiLSTM-AE)
*   **PyTorch 1.8+** (Comparison Models)
*   NVIDIA GPU (CUDA 지원 권장)

### 3.2. 설치
```bash
git clone <repository_url>
cd HAI-Anomaly-Detection
pip install -r requirements.txt
```

### 3.3. 데이터셋 준비
*   **HAI 23.05** 데이터셋을 다운로드하여 `data/hai-23.05/` 경로에 위치시킴.
    *   Github: [https://github.com/icsdataset/hai](https://github.com/icsdataset/hai)

---

## 4. 제안 모델 실험 (Proposed Model)

논문에서 제안하는 **ConvBiLSTM-AE** 및 Baseline 모델(Linear, CAE, LSTM, BiLSTM)을 학습하고 평가함.

### 4.1. 학습 (`train.py`)
```bash
# ConvBiLSTM-AE 학습 (기본 설정)
python train.py --model Conv_BiLSTM_AE --epochs 60

# 다른 Baseline 모델 학습 예시
python train.py --model BiLSTM_AE
```

### 4.2. 평가 (`evaluate.py`)
학습된 모델을 로드하여 Test 1, Test 2 데이터셋에 대한 이상 탐지 성능(F1, AUC, Precision, Recall)을 측정함.
```bash
python evaluate.py
```

### 4.3. 분석 (`analysis_eda.py`)
데이터의 상관관계(Correlation Heatmap) 분석 및 학습된 인코더의 Latent Vector에 대한 **PCA 시각화**를 수행함.
```bash
python analysis_eda.py
```

---

## 5. 비교 모델 실험 (Benchmarks)

최신 SOTA 모델들과의 성능 비교를 위해 PyTorch 기반의 구현체를 제공함. 모든 비교 모델은 `utils/preprocessing.py`를 통해 전처리된 데이터를 공통으로 사용함.

| 모델명 | 특징 | 실행 명령어 |
| :--- | :--- | :--- |
| **MTAD-GAT** | Graph Attention Network 기반 시공간 상관관계 모델링 | `python comparisons/train_mtad_gat.py` |
| **OmniAnomaly** | Stochastic RNN (GRU-VAE) 기반 확률적 모델링 | `python comparisons/train_omni_anomaly.py` |
| **TranAD** | Transformer + Adversarial Training 기반 | `python comparisons/train_tran_ad.py` |
| **USAD** | AutoEncoder + GAN (Adversarial Training) | `python comparisons/train_usad.py` |
| **DAEMON** | Adversarial AE (Reconstruction & Latent Discriminator) | `python comparisons/train_daemon.py` |
| **MAD-GAN** | LSTM-GAN 기반 (Latent Space Optimization 포함) | `python comparisons/train_madgan.py` |

> **참고**: 모든 비교 모델 실험 결과는 `data/outputs/[모델명]/` 폴더에 저장됨.

---

## 6. 인용 (Citation)

본 코드를 연구에 활용할 경우, 아래 논문을 인용 바람.

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

## 7. 특허 (Patent)

본 연구 결과물은 대한민국 특허청에 출원됨.

*   **발명의 명칭**: 다변수 산업 사물 단말 관련 시계열 데이터에서 딥 러닝 모델을 기초로 한, 이상 탐지 방법 그 장치
    *   (METHOD FOR ANOMALY DETECTING BASED ON DEEP LEARNING MODEL IN TIME SERIES DATA RELATED TO MULTIVARIATE INDUSTRIAL THINGS TERMINALS, AND APPARATUS THEREOF)
*   **출원 번호**: 10-2024-0161756
*   **출원 일자**: 2024.11.14
*   **출원인**: 중앙대학교 산학협력단
*   **발명자**: 김상민, 이병천, 문지훈, 노승민, 무아잠 마쿠수드

---

## 8. 라이선스 (License)

이 프로젝트는 **Creative Commons Attribution 4.0 International License (CC BY 4.0)**에 따라 라이선스가 부여됨.

This work is licensed under a Creative Commons Attribution 4.0 International License.

*   **저작권자**: Byeongcheon Lee, Sangmin Kim, Muazzam Maqsood, Jihoon Moon, Seungmin Rho
*   **출처**: Computers, Materials & Continua (CMC), 2024, vol.81, no.1.

단, `comparisons/` 폴더 내의 각 SOTA 비교 모델 코드는 해당 원본 논문 및 저자들의 라이선스 정책을 따름. 각 모델의 원본 리포지토리를 참조 바람.