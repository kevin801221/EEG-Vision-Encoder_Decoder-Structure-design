# EEG-Vision-Encoder_Decoder-Structure-design
![image](https://github.com/user-attachments/assets/60df464d-93f9-44cd-aa28-83c46a99ecde)

為了實現EEG的視覺解碼與生成，本項目建立了一個完整的深度學習框架，包含以下主要功能：

# EEG 視覺解碼與重建框架 - 自定義訓練指南

基於 EEG 的視覺解碼與重建框架，本指南將詳細說明如何使用自己的數據集進行訓練。

# EEG 視覺解碼與重建框架

基於腦電圖（EEG）的端到端視覺解碼與重建框架，整合 CLIP、VAE 和語義管道的多模態方法。

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](arxiv_link)
[![Conference](https://img.shields.io/badge/Conference-NeurIPS%202024-blue)](neurips_link)
[![License](https://img.shields.io/badge/License-MIT-green)](license_link)

## 目錄
- [功能特點](#功能特點)
- [快速開始](#快速開始)
- [環境配置](#環境配置)
- [數據準備](#數據準備)
- [模型訓練](#模型訓練)
- [結果評估](#結果評估)
- [常見問題](#常見問題)

## 功能特點

### 主要特點
- 🧠 多模態特徵提取
- 🎯 高低級特徵融合
- 📈 多管道協同工作
- 🔄 端到端訓練流程

### 技術優勢
- 高時間分辨率
- 低成本實現
- 優秀的泛化性能
- 豐富的視覺重建效果

## 快速開始

### 1. 獲取專案
```bash
git clone https://github.com/{username}/EEG_Image_decode.git
cd EEG_Image_decode
```

### 2. 環境配置
```bash
# 方法一：自動配置
. setup.sh
conda activate BCI

# 方法二：手動配置
conda env create -f environment.yml
conda activate BCI

# 安裝依賴
pip install wandb einops open_clip_torch
pip install transformers==4.28.0.dev0
pip install diffusers==0.24.0
pip install braindecode==0.8.1
```

### 3. 數據準備
```bash
# 創建目錄結構
mkdir -p project_directory/eeg_dataset/raw_data
mkdir -p project_directory/eeg_dataset/preprocessed_data
mkdir -p project_directory/image_set
```

## 數據預處理

### 1. EEG 數據預處理流程
```python
# 1. 濾波處理
def preprocess_eeg(raw_data):
    # 帶通濾波 (0.5-45Hz)
    filtered = filter_data(raw_data, sfreq=1000, 
                         l_freq=0.5, h_freq=45)
    
    # 去除工頻干擾
    notch_filtered = notch_filter(filtered, 
                                freqs=[50, 60])
    
    # 分段
    epochs = create_epochs(notch_filtered, 
                         tmin=-0.2, tmax=1.0)
    
    # 伪迹去除
    clean = remove_artifacts(epochs)
    
    return clean
```

### 2. 數據格式要求
- EEG 數據：(batch_size, channels, timestamps)
- 圖像數據：224x224 或 256x256 RGB格式
- 時間窗口：與視覺刺激對齊

## 模型訓練

### 1. 檢索任務
```bash
# 單一受試者訓練
cd Retrieval/
python ATMS_retrieval.py \
    --logger True \
    --gpu cuda:0 \
    --output_dir ./outputs/contrast

# 聯合受試者訓練
python ATMS_retrieval_joint_train.py \
    --joint_train \
    --sub sub-01 True \
    --logger True \
    --gpu cuda:0 \
    --output_dir ./outputs/contrast
```

### 2. 重建任務

#### CLIP 管道（高級特徵）
```bash
cd Generation/
python ATMS_reconstruction.py \
    --insubject True \
    --subjects sub-08 \
    --logger True \
    --gpu cuda:0 \
    --output_dir ./outputs/contrast
```

**工作原理：**
- EEG信號 → CLIP特徵空間映射
- 特徵空間 → 條件擴散生成
- 生成高質量重建圖像

#### VAE 管道（低級特徵）
```bash
# 步驟1：訓練VAE
python train_vae_latent_512_low_level_no_average.py

# 步驟2：重建圖像
jupyter notebook 1x1024_reconstruct_sdxl.ipynb
```

**工作原理：**
- 提取低級視覺特徵
- 保留紋理和形狀信息
- 重建細節特徵

#### 語義管道
```bash
# 三步驟執行
jupyter notebook image_adapter.ipynb
jupyter notebook GIT_caption_batch.ipynb
jupyter notebook 1x1024_reconstruct_sdxl.ipynb
```

## 結果評估

### 1. 評估指標
```bash
jupyter notebook Reconstruction_Metrics_ATM.ipynb
```

評估內容：
- 圖像質量（PSNR、SSIM）
- 語義相似度（CLIP Score）
- 感知質量（FID Score）

### 2. 可視化結果
- 原始圖像對比
- 特徵圖可視化
- 重建質量分析

## 常見問題

### 1. 環境配置問題
Q: conda 創建環境失敗？
A: 嘗試以下解決方案：
```bash
conda update -n base conda
conda clean --all
```

### 2. 數據處理問題
Q: 數據格式不匹配？
A: 確保數據格式如下：
```python
print(eeg_data.shape)  # (batch_size, channels, timestamps)
print(image_data.shape)  # (batch_size, 3, height, width)
```

### 3. 訓練問題
Q: 顯存不足？
A: 調整參數：
```bash
# 減小批次大小
python ATMS_retrieval.py --batch_size 32

# 使用梯度累積
python ATMS_retrieval.py --gradient_accumulation_steps 4
```

## 使用建議

### 1. 入門階段
1. 使用預處理數據
2. 從單一受試者開始
3. 使用默認參數

### 2. 進階使用
1. 嘗試不同編碼器
2. 調整模型參數
3. 組合多個管道

## 引用

如果您使用了本項目的代碼或方法，請引用我們的論文：
```bibtex
@article{li2024visual,
  title={Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion},
  author={Li, Dongyang and Wei, Chen and Li, Shiying and Zou, Jiachen and Liu, Quanying},
  journal={arXiv preprint arXiv:2403.07721},
  year={2024}
}
```

## 維護者

- 項目負責人：[姓名](mailto:email@example.com)
- 技術支持：[姓名](mailto:email@example.com)

## 參考資源

- [原始論文](paper_link)
- [數據預處理指南](preprocessing_link)
- [模型架構詳解](architecture_link)
- [訓練技巧](training_tips_link)

## 維護與更新

本項目持續更新中，如有問題請提交 Issue 或 Pull Request。


如有任何問題，請聯繫：kilong31442@gmail.com
