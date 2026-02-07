import json
import cv2
import numpy as np
from PIL import Image
import torch
import random
import os

from segmentation.sam3.model_builder import build_sam3_image_model
from segmentation.sam3.model.sam3_image_processor import Sam3Processor

class AnimalEyePipeline:
    def __init__(self, json_path, image_root, sam3_ckpt, output_json=None, show=True):
        """
        :param output_json: 如果為 None，則直接覆蓋原本的 json_path
        :param show:   控制是否顯示處理過程的視窗 (True/False)
        """
        self.json_path = json_path
        self.image_root = image_root
        self.sam3_ckpt = sam3_ckpt
        
        # 設定是否顯示圖片
        self.show = show
        
        # 設定輸出路徑：如果沒指定，就用原本的路徑 (原地更新)
        self.output_json = output_json if output_json else json_path
        
        self.class_colors = {}
        self.eye_color = (0, 255, 0)

        print(f"[INFO] 讀取 {self.json_path} ...")
        self.data = self._load_json()
        
        print(f"[INFO] 載入 SAM3 ({self.sam3_ckpt}) ...")
        self.model = build_sam3_image_model(checkpoint_path=self.sam3_ckpt, load_from_HF=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Sam3Processor(self.model, device = device)

    def _load_json(self):
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"找不到 JSON: {self.json_path}")
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_class_color(self, cls_name):
        if cls_name not in self.class_colors:
            self.class_colors[cls_name] = tuple([random.randint(50, 255) for _ in range(3)])
        return self.class_colors[cls_name]

    def _overlay_mask(self, image, mask, color, alpha=0.5):
        # 如果不顯示圖片，其實可以跳過這裡的運算來加速，但為了保持邏輯簡單，保留運算
        mask_bool = mask > 0
        if not mask_bool.any(): return image
        image[mask_bool] = (image[mask_bool] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        return image

    def _process_yolo_mask(self, vis_img, obj, color):
        if "segmentation" not in obj or not obj["segmentation"]: return
        try:
            mask_yolo = np.zeros(vis_img.shape[:2], dtype=np.uint8)
            seg = obj["segmentation"]
            if isinstance(seg, str): seg = json.loads(seg)
            pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask_yolo, [pts], 255)
            self._overlay_mask(vis_img, mask_yolo, color, alpha=0.3)
        except Exception as e:
            # print(f"  [WARN] YOLO mask 解析失敗: {e}")
            pass

    def _process_sam_eyes(self, img_bgr, vis_img, obj, obj_index, img_name, all_objects):
        x1, y1, x2, y2 = map(int, obj["bbox"])
        cls_name = obj["class"]

        # 1. 互斥遮罩
        clean_img = img_bgr.copy()
        for other_idx, other_obj in enumerate(all_objects):
            if other_idx == obj_index: continue
            if "segmentation" in other_obj and other_obj["segmentation"]:
                try:
                    seg = other_obj["segmentation"]
                    if isinstance(seg, str): seg = json.loads(seg)
                    pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(clean_img, [pts], (0, 0, 0))
                except: pass

        # 2. SAM3 推論
        masked_input = np.zeros_like(clean_img)
        masked_input[y1:y2, x1:x2] = clean_img[y1:y2, x1:x2]
        
        image_pil = Image.fromarray(cv2.cvtColor(masked_input, cv2.COLOR_BGR2RGB))
        state = self.processor.set_image(image_pil)
        output = self.processor.set_text_prompt(state=state, prompt="eyes")
        masks = output["masks"]

        if masks is None or len(masks) == 0:
            print(f"  [FAIL] obj {obj_index} ({cls_name}) 沒偵測到 eyes")
            return []

        # 3. 收集並轉字串
        eyes_segmentation_strings = [] 
        for mask in masks:
            mask_binary = mask.cpu().numpy().squeeze().astype(np.uint8)
            mask_uint8 = (mask_binary > 0).astype(np.uint8) * 255
            
            # 只有當需要視覺化時，才真的去畫圖
            if self.show:
                self._overlay_mask(vis_img, mask_binary, self.eye_color, alpha=0.6)
            
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                contour_points = cnt.reshape(-1, 2).tolist()
                if len(contour_points) > 3:
                    seg_string = json.dumps(contour_points)
                    eyes_segmentation_strings.append(seg_string)

        count = len(eyes_segmentation_strings)
        print(f"[SUCCESS] obj {obj_index} ({cls_name}) 新增 {count} 筆 eyes 資料")
        return eyes_segmentation_strings

    def save_results(self):
        """將結果寫回指定的 JSON 檔案 (預設為覆蓋原檔)"""

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                    np.int16, np.int32, np.int64, np.uint8,
                                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                return super(NumpyEncoder, self).default(obj)

        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

        print(f"[SAVE] 更新檔案: {self.output_json} ...")

    def run(self):
        try:
            for img_name, record in self.data.items():
                img_path = os.path.join(self.image_root, img_name)
                img_bgr = cv2.imread(img_path)
                
                if img_bgr is None:
                    print(f"[ERROR] 無法讀取: {img_path}")
                    continue

                print(f"\n[PROCESSING] {img_name} ...")
                
                # 如果不需要視覺化，這裡其實可以不複製圖片以節省記憶體，
                # 但為了邏輯一致性，我們還是保留，但只在 show=True 時顯示。
                vis = img_bgr.copy() if self.show else None
                
                all_objects = record.get("objects", [])
                is_modified = False 

                for i, obj in enumerate(all_objects):
                    cls = obj["class"]
                    
                    # 只有在要顯示時才畫 YOLO mask
                    if self.show and vis is not None:
                        color = self._get_class_color(cls)
                        self._process_yolo_mask(vis, obj, color)

                    # [關鍵邏輯] 檢查是否已經有 eyes 資料
                    if "eyes" in obj and obj["eyes"]:
                        print(f"  [SKIP] obj {i} ({cls}) 已有 eyes 資料，跳過。")
                        continue
                    
                    # 執行 SAM3 (如果 vis 是 None，內部會略過畫圖)
                    eyes_data = self._process_sam_eyes(img_bgr, vis, obj, i, img_name, all_objects)
                    
                    obj["eyes"] = eyes_data 
                    is_modified = True

                # 只有當：1. 有更動 且 2. 設定開啟視覺化 時，才顯示視窗
                if is_modified and self.show and vis is not None:
                    cv2.imshow("Incremental Pipeline", vis)
                    key = cv2.waitKey(1) 
                    if key == 27: 
                        print("[EXIT] 使用者中斷。")
                        break
            
        except KeyboardInterrupt:
            print("\n[STOP] 使用者強制停止，正在儲存目前進度...")
        
        finally:
            if self.show:
                cv2.destroyAllWindows()
            # 無論正常結束或報錯中斷，都嘗試儲存結果
            self.save_results()

# ================= 執行區 =================
if __name__ == "__main__":

    """
    input: animal_masks.json (來自 YOLO (bbox、segmentation) 的輸出)
    output: animal_masks.json (直接覆蓋，新增 "eyes" 欄位)
    """
    CONFIG = {
        "json_path": "animal_masks.json",     # 輸入檔案
        "image_root": "animal_images",
        "sam3_ckpt": "segmentation/sam3/checkpoints/sam3.pt",
        "output_json": None,
        
        # [控制點] True: 跳出視窗顯示結果, False: 背景執行不顯示
        "show": False  
    }

    pipeline = AnimalEyePipeline(**CONFIG)
    pipeline.run()