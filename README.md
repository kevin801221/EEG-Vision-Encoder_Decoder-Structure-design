# EEG-Vision-Encoder_Decoder-Structure-design
![image](https://github.com/user-attachments/assets/476eb377-45b4-4f25-b553-af4b8ae4e47c)

![image](https://github.com/user-attachments/assets/726d9b9f-1d17-488e-ab13-f1efb5383618)

# EEG 視覺解碼與重建框架

一個基於腦電圖（EEG）的端到端視覺重建零樣本框架，使用自適應思維映射器（ATM）將神經信號轉換為視覺重建。

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](arxiv_link)
[![Conference](https://img.shields.io/badge/Conference-NeurIPS%202024-blue)](neurips_link)
[![License](https://img.shields.io/badge/License-MIT-green)](license_link)

## 目錄
- [框架概述](#框架概述)
- [快速開始](#快速開始)
- [詳細步驟](#詳細步驟)
- [結果評估](#結果評估)
- [常見問題](#常見問題)

## 框架概述

本框架將 EEG 視覺重建分為兩個關鍵階段：
1. 特徵提取階段：使用 ATM 提取多層次特徵
2. 圖像生成階段：整合特徵重建最終圖像

### 主要特點
- 🧠 自適應思維映射技術
- 🎯 多路徑並行處理
- 📈 特徵融合生成
- 🔄 零樣本遷移能力

### 技術優勢
- 低成本實現
- 高時間分辨率
- 優秀的泛化性能
- 多模態支持

## 快速開始

### 1. 環境配置
```bash
# 克隆專案
git clone https://github.com/{username}/EEG_Image_decode.git
cd EEG_Image_decode

# 配置環境
conda env create -f environment.yml
conda activate BCI

# 安裝依賴
pip install -r requirements.txt
```

### 2. 數據準備
```bash
# 創建必要目錄
mkdir -p data/{raw,preprocessed,output}

# 下載示例數據（如果需要）
python scripts/download_data.py
```

## 詳細步驟

### 第一階段：特徵提取

#### 1. EEG 預處理
```bash
# 運行預處理腳本
python preprocessing/eeg_preprocess.py \
    --input_dir data/raw \
    --output_dir data/preprocessed \
    --sampling_rate 1000
```

#### 2. ATM 特徵提取
```bash
# 運行 ATM 提取三路特徵
python feature_extraction/run_atm.py \
    --input_data data/preprocessed \
    --output_dir data/features \
    --batch_size 32 \
    --gpu cuda:0
```

這一步會產生三種特徵：
- CLIP 特徵（高級語義）
- 擴散先驗（結構信息）
- VAE 特徵（視覺細節）

### 第二階段：圖像重建

#### 1. 特徵整合
```bash
# 整合第一階段的特徵
python generation/integrate_features.py \
    --clip_features data/features/clip \
    --prior_features data/features/prior \
    --vae_features data/features/vae \
    --output_dir data/features/integrated
```

#### 2. 圖像生成
```bash
# 使用整合特徵生成最終圖像
python generation/generate_images.py \
    --input_features data/features/integrated \
    --model_type sdxl \
    --output_dir data/output/final
```

## 模型結構

### ATM 編碼器
```python
class ATMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.eeg_encoder = EEGEncoder()
        self.clip_projector = CLIPProjector()
        self.vae_encoder = VAEEncoder()
        self.diffusion = DiffusionModel()
```

### 特徵整合器
```python
class FeatureIntegrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic_head = SemanticHead()
        self.structure_head = StructureHead()
        self.detail_head = DetailHead()
```

## 數據流程說明

### 第一階段到第二階段的數據流動：

1. **CLIP 路徑**
   - 輸入：EEG 信號
   - 處理：ATM → CLIP 投影
   - 輸出：高級語義特徵
   - 用途：指導圖像內容生成

2. **擴散路徑**
   - 輸入：EEG 信號
   - 處理：ATM → 擴散模型
   - 輸出：圖像先驗
   - 用途：提供結構約束

3. **VAE 路徑**
   - 輸入：EEG 信號
   - 處理：ATM → VAE
   - 輸出：模糊圖像
   - 用途：提供視覺細節

### 特徵整合過程
```python
def integrate_features(semantic, structure, detail):
    # 特徵對齊
    aligned_features = align_features(semantic, structure, detail)
    
    # 特徵融合
    fused_features = feature_fusion(aligned_features)
    
    return fused_features
```

## 參數配置

### 第一階段參數
```yaml
# config/stage1_config.yaml
atm:
  input_channels: 64
  sampling_rate: 1000
  feature_dim: 512

clip:
  model_type: "ViT-L/14"
  projection_dim: 768

vae:
  latent_dim: 512
  recon_weight: 1.0
```

### 第二階段參數
```yaml
# config/stage2_config.yaml
diffusion:
  model: "sdxl"
  steps: 50
  guidance_scale: 7.5

integration:
  semantic_weight: 1.0
  structure_weight: 0.8
  detail_weight: 0.5
```

## 結果評估

### 1. 評估指標
```bash
# 運行評估腳本
python evaluate/run_metrics.py \
    --generated_images data/output/final \
    --ground_truth data/test/images \
    --output_dir data/evaluation
```

### 2. 可視化結果
```bash
# 生成可視化報告
python visualize/create_report.py \
    --results_dir data/evaluation \
    --output_path reports/evaluation.html
```

## 常見問題

### 1. 特徵提取問題
Q: 三個特徵路徑是否必須同時使用？
A: 不是必須的，但完整使用三個路徑能獲得最佳效果。每個路徑負責不同層面的視覺重建。

### 2. 整合問題
Q: 特徵整合失敗怎麼辦？
A: 檢查以下幾點：
- 特徵維度是否匹配
- 特徵範圍是否正確歸一化
- 權重配置是否合理

### 3. 性能優化
Q: 如何提升重建質量？
A: 可以：
- 調整特徵權重
- 增加訓練數據
- 優化特徵提取參數

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

## 許可證

本項目基於 MIT 許可證開源。

## 維護者

- 技術支持：[姓名](mailto:email@example.com)

## 更新日誌
- [2024/09/26] 論文被 NeurIPS 2024 接收
- [2024/09/25] 更新 arXiv 論文
- [2024/08/01] 更新訓練和推理腳本
- [2024/05/19] 更新數據集加載腳本
- [2024/03/12] 發布 arXiv 論文
