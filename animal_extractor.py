import os
import json
from ultralytics import YOLO
import numpy as np

class AnimalDetector:
    def __init__(self, input_folder, output_json, model_path="yolo11l-seg.pt"):
        """
        初始化產生器
        :param input_folder: 圖片來源資料夾
        :param output_json: 輸出的 JSON 檔名
        :param model_path: YOLO 模型路徑
        """
        self.input_folder = input_folder
        self.output_json = output_json
        self.model_path = model_path
        
        # COCO 動物類別索引 (鳥, 貓, 狗, 馬, 羊, 牛, 象, 熊, 斑馬, 長頸鹿)
        self.animal_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        self.model = None

    def load_model(self):
        """載入 YOLO 模型"""
        print(f"[INFO] 載入模型 {self.model_path}...")
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"模型載入失敗: {e}")
            raise

    def _get_image_files(self):
        """取得資料夾內所有圖片檔案"""
        if not os.path.exists(self.input_folder):
            print(f"資料夾不存在: {self.input_folder}")
            return []
        
        return [
            f for f in os.listdir(self.input_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]

    def _format_segmentation(self, contour_points):
        """
        將輪廓點轉成緊湊的 JSON 字串格式
        Format: "[[x1,y1],[x2,y2]...]"
        """
        return json.dumps(
            contour_points.tolist(),
            separators=(',', ':') # 去除空格，讓字串更短
        )

    def run(self):
        """執行主要流程：讀檔 -> 推論 -> 存檔"""
        if self.model is None:
            self.load_model()

        image_files = self._get_image_files()
        if not image_files:
            return

        print(f"[INFO] 找到 {len(image_files)} 張圖片，開始處理...")
        all_records = {}

        # 3. 批次處理
        for img_file in image_files:
            img_path = os.path.join(self.input_folder, img_file)
            
            # YOLO 推論
            results = self.model(img_path, classes=self.animal_classes, verbose=False)[0]
            
            img_data = []
            
            # 判斷是否有偵測到物件 (Mask)
            if results.masks is not None:
                h, w = results.orig_shape
                
                # 遍歷偵測到的物件
                for i, contour_points in enumerate(results.masks.xy):
                    # 取得基本資訊
                    cls_id = int(results.boxes.cls[i])
                    class_name = self.model.names[cls_id]
                    conf = float(results.boxes.conf[i])
                    
                    # 取得 BBox 並轉為 int list
                    box = results.boxes.xyxy[i].cpu().numpy().astype(int).tolist()

                    # 處理 Segmentation (轉字串)
                    seg_str = self._format_segmentation(contour_points)

                    img_data.append({
                        "object_id": i,
                        "class": class_name,
                        "confidence": round(conf, 4),
                        "bbox": box,
                        "segmentation": seg_str,
                        "eyes": [] # 預留欄位給後續 SAM3 使用
                    })

                # 記錄該張圖片的結果
                all_records[img_file] = {
                    "width": w,
                    "height": h,
                    "animals_found": len(img_data),
                    "objects": img_data
                }
            else:
                # 沒偵測到動物，但還是記錄圖片資訊
                # (如果你希望沒動物就不存，可以在這裡改邏輯)
                h, w = results.orig_shape
                all_records[img_file] = {
                    "width": w,
                    "height": h,
                    "animals_found": 0,
                    "objects": []
                }

            print(f"{img_file}：發現 {len(img_data)} 隻動物")

        # 4. 存檔
        self.save_json(all_records)

    def save_json(self, data):
        """將結果寫入 JSON"""
        print(f"[SAVE] 更新檔案 {self.output_json}...")
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


# ================= 使用範例 =================
if __name__ == "__main__":
    # 設定參數
    CONFIG = {
        "input_folder": "animal_images",
        "output_json": "animal_masks.json",
        "model_path": "yolo11l-seg.pt"
    }

    # 初始化並執行
    detector = AnimalDetector(**CONFIG)
    detector.run()