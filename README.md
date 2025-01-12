# EEG-Vision-Encoder_Decoder-Structure-design
![image](https://github.com/user-attachments/assets/60df464d-93f9-44cd-aa28-83c46a99ecde)

為了實現EEG的視覺解碼與生成，本項目建立了一個完整的深度學習框架，包含以下主要功能：

# EEG 視覺解碼與重建框架 - 自定義訓練指南

基於 EEG 的視覺解碼與重建框架，本指南將詳細說明如何使用自己的數據集進行訓練。

## 目錄
- [專案結構](#專案結構)
- [環境配置](#環境配置)
- [數據集準備](#數據集準備)
- [自定義訓練流程](#自定義訓練流程)
- [模型配置與修改](#模型配置與修改)
- [常見問題解答](#常見問題解答)

## 專案結構

```plaintext
your-project/
├── data/
│   ├── raw_data/              # 原始數據
│   │   ├── eeg/              # EEG 記錄
│   │   └── images/           # 刺激圖像
│   │
│   ├── preprocessed_data/     # 預處理後的數據
│   │   ├── eeg_features/     # EEG 特徵
│   │   ├── image_features/   # 圖像特徵
│   │   └── metadata/         # 標註信息
│   │
│   └── splits/               # 數據集劃分
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── models/
│   ├── encoders/             # 編碼器模型
│   │   ├── eeg_encoder.py
│   │   └── image_encoder.py
│   │
│   ├── decoders/            # 解碼器模型
│   └── configs/             # 模型配置文件
│
├── scripts/
│   ├── preprocess/          # 預處理腳本
│   ├── train/              # 訓練腳本
│   └── evaluate/           # 評估腳本
│
├── outputs/                 # 輸出目錄
│   ├── checkpoints/        # 模型檢查點
│   ├── logs/              # 訓練日誌
│   └── results/           # 結果輸出
│
├── environment.yml         # 環境配置文件
└── setup.sh               # 環境安裝腳本
```

## 環境配置

### 1. 基礎環境設置
```bash
# 創建並激活環境
conda env create -f environment.yml
conda activate eeg_decode

# 安裝其他依賴
pip install -r requirements.txt
```

### 2. CUDA 配置
確保您已安裝適配的 CUDA 版本：
```bash
# 檢查 CUDA 版本
nvidia-smi

# 如果需要特定版本的 PyTorch
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## 數據集準備

### 1. 數據格式要求

#### EEG 數據格式
```python
# EEG 數據格式示例
{
    'subject_id': 'sub-01',
    'trial_id': 'trial-001',
    'eeg_data': np.array([channels, timestamps]),  # shape: (64, 1000) 
    'sampling_rate': 1000,
    'stimulus_onset': 0,
    'stimulus_offset': 500
}
```

#### 圖像數據格式
- 支持的格式：JPG, PNG
- 建議尺寸：224x224 或 256x256
- 顏色空間：RGB

### 2. 數據預處理步驟

1. **EEG 數據預處理**
```bash
# 1. 將原始 EEG 數據放入相應目錄
mkdir -p data/raw_data/eeg
cp your_eeg_data/* data/raw_data/eeg/

# 2. 運行預處理腳本
python scripts/preprocess/eeg_preprocess.py \
    --input_dir data/raw_data/eeg \
    --output_dir data/preprocessed_data/eeg_features \
    --sampling_rate 1000 \
    --channels 64
```

2. **圖像預處理**
```bash
# 1. 準備刺激圖像
mkdir -p data/raw_data/images
cp your_images/* data/raw_data/images/

# 2. 運行圖像預處理
python scripts/preprocess/image_preprocess.py \
    --input_dir data/raw_data/images \
    --output_dir data/preprocessed_data/image_features \
    --size 224
```

3. **創建數據集劃分**
```python
# 在 scripts/preprocess/create_splits.py 中配置劃分比例
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 運行劃分腳本
python scripts/preprocess/create_splits.py \
    --data_dir data/preprocessed_data \
    --output_dir data/splits
```

## 自定義訓練流程

### 1. 修改配置文件
在 `models/configs/` 中創建自定義配置：

```yaml
# models/configs/custom_config.yaml
model:
  eeg_encoder:
    type: 'EEGNetv4'  # 或其他支持的編碼器
    input_channels: 64
    sampling_rate: 1000
    dropout_rate: 0.5
    
  image_encoder:
    type: 'ResNet50'
    pretrained: true
    
training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  early_stopping_patience: 10
```

### 2. 啟動訓練

#### 檢索任務訓練
```bash
# 單subject訓練
python scripts/train/train_retrieval.py \
    --config models/configs/custom_config.yaml \
    --data_dir data/preprocessed_data \
    --splits_dir data/splits \
    --output_dir outputs/retrieval \
    --subject sub-01 \
    --gpu cuda:0

# 聯合訓練
python scripts/train/train_retrieval_joint.py \
    --config models/configs/custom_config.yaml \
    --data_dir data/preprocessed_data \
    --splits_dir data/splits \
    --output_dir outputs/retrieval_joint
```

#### 重建任務訓練
```bash
# CLIP 管道訓練
python scripts/train/train_reconstruction.py \
    --config models/configs/custom_config.yaml \
    --pipeline clip \
    --data_dir data/preprocessed_data \
    --output_dir outputs/reconstruction_clip

# 低級特徵管道訓練
python scripts/train/train_reconstruction.py \
    --config models/configs/custom_config.yaml \
    --pipeline low_level \
    --data_dir data/preprocessed_data \
    --output_dir outputs/reconstruction_low
```

## 模型配置與修改

### 1. EEG 編碼器選項

可以在 `models/encoders/eeg_encoder.py` 中修改或添加編碼器：

```python
# 支持的編碼器類型
encoders = {
    'EEGNetv4': EEGNetv4Encoder,
    'DeepConvNet': DeepConvNetEncoder,
    'ShallowConvNet': ShallowConvNetEncoder,
    'Custom': CustomEncoder
}

# 編碼器參數示例
encoder_params = {
    'EEGNetv4': {
        'F1': 8,  # 時間卷積核數量
        'D': 2,   # 空間卷積核數量
        'F2': 16, # 可分離卷積核數量
        'dropout_rate': 0.5
    }
}
```

### 2. 自定義編碼器

創建新的編碼器類：

```python
# models/encoders/custom_encoder.py
class CustomEEGEncoder(nn.Module):
    def __init__(self, input_channels=64, sampling_rate=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 51), padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        # 添加更多層...

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # 處理流程...
        return x
```

## 常見問題解答

### 1. 內存問題
Q: 訓練時出現 OOM (內存溢出) 錯誤？
A: 嘗試以下解決方案：
- 減小 batch_size
- 使用梯度累積
- 縮小模型規模

### 2. 數據格式問題
Q: EEG 數據格式不匹配？
A: 確保數據格式如下：
```python
# 檢查數據格式
print(eeg_data.shape)  # 應該是 (batch_size, channels, timestamps)
print(image_data.shape)  # 應該是 (batch_size, 3, height, width)
```

### 3. 訓練問題
Q: 訓練不收斂？
A: 檢查以下幾點：
- 學習率設置
- 數據預處理
- 模型架構
- 損失函數選擇

## 參考資源

- [原始論文](paper_link)
- [數據預處理指南](preprocessing_link)
- [模型架構詳解](architecture_link)
- [訓練技巧](training_tips_link)

## 維護與更新

本項目持續更新中，如有問題請提交 Issue 或 Pull Request。


如有任何問題，請聯繫：kilong31442@gmail.com
