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

## 🚀 本地運行步驟

### 環境準備
1. 克隆專案：
   ```bash
   git clone [https://github.com/ChengShaoFong/sam3-eyes-dist-measure.git](https://github.com/ChengShaoFong/sam3-eyes-dist-measure.git)
   cd sam3-eyes-dist-measure

# 1. 建立並啟動 Conda 環境
conda create -n sfc python=3.10
conda activate sfc

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 執行主程式
python main.py --input data/samples --output output_results
