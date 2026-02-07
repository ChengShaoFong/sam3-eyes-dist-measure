import json
import csv
import os

class TestVerifier:
    def __init__(self, ground_truth_path, system_json_path):
        """
        :param ground_truth_path: 人工標註的 CSV 檔案路徑。
        :param system_json_path: 系統產出的 JSON 檔案路徑 (animal_masks.json)。
        """
        self.gt_path = ground_truth_path
        self.sys_json_path = system_json_path
        
        # 用於儲存從 JSON 解析出的系統數據
        self.sys_individual_map = {} # (img_name, obj_id) -> eyes_dist
        self.sys_pair_map = {}       # (img_name, sorted_pair_ids) -> right_eye_dist

    def _load_gt_csv(self):
        """讀取標準答案 CSV。"""
        if not os.path.exists(self.gt_path):
            print(f"找不到標準答案檔案: {self.gt_path}")
            return []
        with open(self.gt_path, 'r', encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))

    def _load_system_json(self):
        """讀取並解析系統輸出的 JSON 資料結構。"""
        if not os.path.exists(self.sys_json_path):
            print(f"找不到系統 JSON 檔案: {self.sys_json_path}")
            return False

        with open(self.sys_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for img_name, record in data.items():
            # 解析單體瞳距 (Individual PD)
            for obj in record.get("objects", []):
                obj_id = str(obj.get("object_id"))
                dist = obj.get("eyes_dist")
                if dist is not None:
                    self.sys_individual_map[(img_name, obj_id)] = float(dist)

            # 解析配對距離 (Pair RR)
            for pair in record.get("pairs", []):
                id1 = str(pair.get("obj_1_id"))
                id2 = str(pair.get("obj_2_id"))
                sorted_ids = tuple(sorted([id1, id2]))
                dist = pair.get("right_eye_dist")
                if dist is not None:
                    self.sys_pair_map[(img_name, sorted_ids)] = float(dist)
        
        return True

    def run(self):
        print("\n" + "="*80)
        print("核心系統誤差分析 (JSON 基底比對)")
        print(f"基準檔案: {self.gt_path}")
        print(f"測試檔案: {self.sys_json_path}")
        print("="*80)

        # 1. 載入資料
        gt_rows = self._load_gt_csv()
        if not self._load_system_json() or not gt_rows:
            return

        total_error = 0.0
        match_count = 0
        miss_count = 0

        # 表頭
        header = f"{'圖片名稱':<18} | {'類型':<10} | {'物件IDs':<10} | {'預期(GT)':<10} | {'實測(Sys)':<10} | {'誤差'}"
        print(header)
        print("-" * 85)

        # 進行比對
        for row in gt_rows:
            img = row['Image_Name']
            dtype = row['Type']
            expect = float(row['Expected_Dist'])
            
            actual = None
            ids_label = ""

            if dtype == 'Individual':
                oid = str(row['Obj_ID_1'])
                ids_label = oid
                actual = self.sys_individual_map.get((img, oid))
            
            elif dtype == 'Pair':
                id1, id2 = str(row['Obj_ID_1']), str(row['Obj_ID_2'])
                sorted_ids = tuple(sorted([id1, id2]))
                ids_label = f"{id1}-{id2}"
                actual = self.sys_pair_map.get((img, sorted_ids))

            # 處理比對結果
            if actual is None:
                print(f"{img[:18]:<18} | {dtype:<10} | {ids_label:<10} | {expect:<10.2f} | {'MISS':<10} | N/A")
                miss_count += 1
                continue

            # 排除異常值 (例如實測為 0 的偵測失敗案例，不計入 MAE)
            if actual == 0 and expect != 0:
                print(f"{img[:18]:<18} | {dtype:<10} | {ids_label:<10} | {expect:<10.2f} | {actual:<10.2f} | 偵測失敗")
                miss_count += 1
                continue

            diff = abs(actual - expect)
            total_error += diff
            match_count += 1

            print(f"{img[:18]:<18} | {dtype:<10} | {ids_label:<10} | {expect:<10.2f} | {actual:<10.2f} | {diff:<10.2f}")

        # 統計報告
        print("-" * 85)
        if match_count > 0:
            mae = total_error / match_count
            print(f"統計分析結果:")
            print(f"  - 成功比對: {match_count} 筆")
            print(f"  - 遺漏/失敗: {miss_count} 筆")
            print(f"  - 平均絕對誤差 (MAE): {mae:.3f} px")
            
            if mae < 5.0:
                print("結果判定: 極度精準 (誤差 < 5px)")
            elif mae < 10.0:
                print("結果判定: 通過 (誤差在合理範圍內)")
            else:
                print("結果判定: 誤差偏高，建議檢查模型參數")
        else:
            print("警告: 沒有找到可供比對的有效數據")

if __name__ == "__main__":
    # 設定檔案路徑
    # 請確保 ground_truth.csv 位於正確位置
    # 請確保 animal_masks.json 位於正確位置 (Stage 3 更新後的版本)
    
    VERIFY_CONFIG = {
        "ground_truth_path": "ground_truth.csv",
        "system_json_path": "animal_masks.json"
    }

    verifier = TestVerifier(**VERIFY_CONFIG)
    verifier.run()