# SAM3 Animal Eyes Distance Measurement System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-green.svg)
![Docker](https://img.shields.io/badge/docker-supported-blue.svg)

## 📖 專案介紹

#### 本專案是一個基於深度學習的自動化動物眼部特徵分析系統。結合 **YOLOv11** 的實例分割能力與 **SAM3 (Segment Anything Model 3)** 的提示詞精準邊緣提取技術，實現對動物瞳距（PD）與相關生理指標的非侵入式精準測量。
---

## 🛠 技術棧說明
- **核心算法**: YOLOv11 (Object Detection), SAM3 (Precision Segmentation)
- **開發語言**: Python 3.10+
- **影像處理**: OpenCV, Pillow, NumPy

---

## 專案準備

```bash
git clone [https://github.com/ChengShaoFong/sam3-eyes-dist-measure.git](https://github.com/ChengShaoFong/sam3-eyes-dist-measure.git)
cd sam3-eyes-dist-measure
```
## 下載模型權重

由於模型權重檔案體積較大，未包含在 Git 倉庫中。執行程式前，請確保已手動下載以下權重檔並放置於指定路徑：

### 1. YOLOv11 Segmentation
* **檔案名稱**：[`yolo11l-seg.pt`](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt)
* **存放路徑**：`./` (專案根目錄)
* **用途**：負責初始影像的實例分割與動物類別偵測。

### 2. SAM3 Checkpoint
* **檔案名稱**：`sam3.pt`
* **存放路徑**：`segmentation/sam3/checkpoints/`
* **用途**：針對眼部特徵進行高精細度的邊緣提取與優化。
  
## Config.ini 配置

本專案使用 `config.ini` 管理所有運行參數，方便快速調整實驗設定：
- **數據準備 (`data_prep`)**: 可設定目標類別 (如 `dog`, `cat`) 及下載限制 以及 自定義test.csv圖片 ( 從指定COCO.json中下載 )。
- **路徑管理 (`paths`)**: 自定義數據存放、遮罩結果 (JSON) 與視覺化輸出路徑。
- **模型載入 (`models`)**: 指定 YOLOv11 與 SAM3 的權重路徑。
- **運行開關 (`flags`)**: 控制是否顯示即時視覺化結果。

## 本地運行步驟

### 1. 建立並啟動 Conda 環境
```
# 1. 建立環境
conda create -n sfc python=3.10
conda activate sfc

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 執行主程式
python main.py 
```

 
## Docker 部署指令
```
# 使用 Docker Compose 進行編譯與啟動
docker compose build
docker compose up -d
```
