import json
import os
import math
import logging
from itertools import combinations

import cv2
import numpy as np

class AnimalMeasurementTool:
    """
    動物測量工具類別：負責解析眼睛座標、計算瞳距 (PD) 與群體間距 (RR)，並產生視覺化結果。
    """
    def __init__(self, json_file_path, image_folder_path, output_folder, show=False):
        self.json_path = json_file_path
        self.image_root = image_folder_path
        self.output_folder = output_folder
        self.show = show
        
        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)

    # ----------------------------------------------------------------------
    # 核心計算邏輯 (私有方法)
    # ----------------------------------------------------------------------

    def _calculate_euclidean_distance(self, p1, p2):
        """計算兩點間的歐幾里得距離。"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _get_centroid(self, seg_data):
        """從多邊形輪廓計算重心。"""
        pts = np.array(json.loads(seg_data) if isinstance(seg_data, str) else seg_data)
        moments = cv2.moments(pts)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            return np.array([cx, cy])
        return np.mean(pts, axis=0).astype(int)

    def _preprocess_coordinates(self, data):
        """解析 eyes 數據並計算重心標記為 L/R。"""
        for img_name, record in data.items():
            for obj in record.get("objects", []):
                eyes = obj.get("eyes", [])
                if len(eyes) == 2:
                    centroids = [self._get_centroid(eye) for eye in eyes]
                    # 以 X 座標排序：視覺左側定義為 eye_L，右側為 eye_R
                    centroids.sort(key=lambda p: p[0])
                    obj["_tmp_eye_L"] = centroids[0]
                    obj["_tmp_eye_R"] = centroids[1]
        return data

    def _calculate_all_metrics(self, data):
        """計算單體 PD 與群體 RR，並清理暫存座標物件。"""
        total_pair_count = 0
        for record in data.values():
            objects = record.get("objects", [])
            objs_with_eyes = []

            # Part A: 單體瞳距 (Pupillary Distance)
            for obj in objects:
                if "_tmp_eye_L" in obj and "_tmp_eye_R" in obj:
                    dist = self._calculate_euclidean_distance(obj["_tmp_eye_L"], obj["_tmp_eye_R"])
                    obj["eyes_dist"] = round(dist, 2)
                    obj["eye_L_center"] = obj["_tmp_eye_L"].tolist()
                    obj["eye_R_center"] = obj["_tmp_eye_R"].tolist()
                    objs_with_eyes.append(obj)

            # Part B: 群體間距 (Right-eye to Right-eye Distance)
            record["pairs"] = []
            for obj_a, obj_b in combinations(objs_with_eyes, 2):
                pair_dist = self._calculate_euclidean_distance(
                    np.array(obj_a["eye_R_center"]), np.array(obj_b["eye_R_center"])
                )
                record["pairs"].append({
                    "obj_1_id": obj_a.get("object_id", "unknown"),
                    "obj_2_id": obj_b.get("object_id", "unknown"),
                    "right_eye_dist": round(pair_dist, 2)
                })
                total_pair_count += 1

            # 清理暫存屬性
            for obj in objects:
                obj.pop("_tmp_eye_L", None)
                obj.pop("_tmp_eye_R", None)

        return data, total_pair_count

    # ----------------------------------------------------------------------
    # 視覺化與輸出 (公用/私有方法)
    # ----------------------------------------------------------------------

    def _apply_visual_annotation(self, img, p1, p2, color, label, offset_y=-10):
        """在影像上繪製線段、圓點與帶背景的數值標籤。"""
        p1, p2 = tuple(map(int, p1)), tuple(map(int, p2))
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

        # 繪製標註線與端點
        cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)
        cv2.circle(img, p1, 4, (0, 0, 255), -1)
        cv2.circle(img, p2, 4, (255, 0, 0), -1)

        # 繪製文字背景與數值
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.45, 1
        (w, h), baseline = cv2.getTextSize(label, font, scale, thick)
        txt_orig = (mid[0] - w // 2, mid[1] + offset_y)
        
        cv2.rectangle(img, (txt_orig[0] - 3, txt_orig[1] - h - 3), 
                      (txt_orig[0] + w + 3, txt_orig[1] + baseline), color, -1)
        cv2.putText(img, label, txt_orig, font, scale, (0, 0, 0), thick, cv2.LINE_AA)

    def draw_frame_results(self, image_path, record):
        """讀取單一影像並繪製所有已計算的 PD 與 RR 資料。"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"無法讀取圖片: {image_path}")
            return None

        objects = record.get("objects", [])
        obj_lookup = {obj.get("object_id"): obj for obj in objects}

        # 1. 繪製單體 PD (黃色)
        for obj in objects:
            if "eyes_dist" in obj:
                self._apply_visual_annotation(
                    img, obj["eye_L_center"], obj["eye_R_center"], 
                    (0, 255, 255), f"PD:{obj['eyes_dist']}"
                )

        # 2. 繪製群體 RR (粉紅色，位移標籤避免重疊)
        for pair in record.get("pairs", []):
            o1 = obj_lookup.get(pair["obj_1_id"])
            o2 = obj_lookup.get(pair["obj_2_id"])
            if o1 and o2 and "eye_R_center" in o1 and "eye_R_center" in o2:
                self._apply_visual_annotation(
                    img, o1["eye_R_center"], o2["eye_R_center"], 
                    (255, 0, 255), f"RR:{pair['right_eye_dist']}", offset_y=20
                )
        return img

    def save_final_json(self, data):
        """將最終運算結果序列化回 JSON 檔案。"""
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        print(f"[SAVE] 更新檔案: {self.json_path}")
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4, cls=CustomEncoder)

    # ----------------------------------------------------------------------
    # 執行
    # ----------------------------------------------------------------------

    def run(self):
        
        if not os.path.exists(self.json_path):
            logging.error(f"找不到指定的 JSON 檔案: {self.json_path}")
            return

        with open(self.json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # 核心計算流程
        data = self._preprocess_coordinates(raw_data)
        processed_data, pair_count = self._calculate_all_metrics(data)
        print(f"[INFO] 共新增 {pair_count} 組群體配對數據。")

        # 視覺化輸出流程
        saved_img_count = 0  # 確保變數名稱統一
        for img_name, record in processed_data.items():
            # 判斷是否具備可視覺化的數據
            has_metric = any("eyes_dist" in o for o in record.get("objects", []))
            
            if has_metric:
                path = os.path.join(self.image_root, img_name)
                vis_frame = self.draw_frame_results(path, record)
                
                if vis_frame is not None:
                    if self.output_folder:
                        out_p = os.path.join(self.output_folder, f"res_{img_name}")
                        cv2.imwrite(out_p, vis_frame)
                        saved_img_count += 1
                    
                    if self.show:
                        cv2.imshow("Animal Measurement Analytics", vis_frame)
                        if cv2.waitKey(0) == 27: break

        if self.show:
            cv2.destroyAllWindows()
            
        
        print(f"[INFO] 共掃描 {len(processed_data)} 張影像，輸出 {saved_img_count} 份視覺化結果。")
        self.save_final_json(processed_data)

if __name__ == "__main__":
    tool = AnimalMeasurementTool(
        json_file_path="animal_masks.json",
        image_folder_path="animal_images",
        output_folder="output_results",
        show=False
    )
    tool.run()
