from pycocotools.coco import COCO
import requests
import os
from tqdm import tqdm
import csv

class AnimalDataSelector:
    def __init__(self, 
                 json_path="data/annotations_trainval2017/instances_train2017.json", 
                 save_folder="animal_images"):
        """
        初始化資料選擇器。
        :param json_path: COCO 標註 JSON 檔案的路徑。
        :param save_folder: 圖片下載後的儲存資料夾。
        """
        self.json_path = json_path
        self.save_folder = save_folder
        self.coco = None
        
        print(f"正在載入 COCO 標註資料庫: {self.json_path} ...")
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"錯誤: 找不到標註檔案，請確認路徑: {self.json_path}")
        
        self.coco = COCO(self.json_path)

    def select_images_from_csv(self, csv_path):
        """
        [模式 A] 從 CSV 檔案中讀取指定圖片清單。
        :param csv_path: 包含檔名的 CSV 檔案路徑。
        :return: COCO 圖片資訊列表 (List of image info dictionaries)。
        """
        print(f"\n[模式 A] 正在讀取指定清單: {csv_path} ...")
        target_ids = []
        
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            rows = csv.reader(f)
            for row in rows:
                if not row: continue
                filename = row[0].strip()
                
                # 解析檔名轉 ID (例如: "000000018155.jpg" -> 18155)
                try:
                    img_id = int(filename.split('.')[0])
                    target_ids.append(img_id)
                except ValueError:
                    print(f"  [警告] 無法解析檔名: {filename}，已跳過。")
        
        if target_ids:
            # 一次載入所有圖片資訊
            images = self.coco.loadImgs(target_ids)
            print(f"  -> 成功解析 {len(images)} 張指定圖片資訊。")
            return images
        else:
            print("  -> CSV 是空的或格式無效。")
            return []

    def select_images_by_category(self, target_categories=['dog', 'cat'], min_count=2, max_images=100):
        """
        [模式 B] 自動篩選包含特定類別的圖片。
        :param target_categories: 目標類別列表 (例如: ['dog', 'cat'])。
        :param min_count: 單張圖片中至少包含多少隻目標動物。
        :param max_images: 最大下載數量限制。
        :return: COCO 圖片資訊列表。
        """
        print(f"\n[模式 B] 正在自動篩選圖片...")
        print(f"  - 目標類別: {target_categories}")
        print(f"  - 最小動物數量: {min_count}")

        # 取得類別 ID
        cat_ids = self.coco.getCatIds(catNms=target_categories)
        # 初步取得包含這些類別的圖片 ID
        img_ids = self.coco.getImgIds(catIds=cat_ids)
        
        selected_images = []
        count = 0
        
        for img_id in img_ids:
            # 檢查該圖片中的標註數量
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
            
            if len(ann_ids) >= min_count:
                img_info = self.coco.loadImgs(img_id)[0]
                selected_images.append(img_info)
                count += 1
                
                # 達到數量限制就停止
                if count >= max_images:
                    break
        
        print(f"  -> 已篩選出 {len(selected_images)} 張符合條件的圖片。")
        return selected_images

    def download_images(self, images_list):
        """
        下載選定的圖片到儲存資料夾。
        :param images_list: COCO 圖片資訊列表。
        """
        if not images_list:
            print("沒有圖片需要下載。")
            return

        print(f"\n開始下載圖片至 '{self.save_folder}/' ...")
        os.makedirs(self.save_folder, exist_ok=True)

        success_count = 0
        for img_info in tqdm(images_list):
            url = img_info['coco_url']
            filename = img_info['file_name']
            filepath = os.path.join(self.save_folder, filename)
            
            # 檢查是否已存在 (快取機制)
            if os.path.exists(filepath):
                success_count += 1
                continue
            
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    success_count += 1
                else:
                    print(f"  [錯誤] 連結無效: {filename} (狀態碼: {response.status_code})")
            except Exception as e:
                print(f"  [錯誤] 下載失敗: {filename} ({e})")

        print(f"\n蒐集到資料夾內共有 {success_count} 張有效圖片。")

    def run(self, csv_path=None, target_categories=['dog', 'cat'], min_count=2, max_images=100):
        """
        主執行流程。
        優先檢查 CSV，若無則執行類別搜尋，最後下載圖片。
        """
        images = []
        
        # 判斷邏輯: 有 CSV 就用 Mode A，否則用 Mode B
        if csv_path and os.path.exists(csv_path):
            images = self.select_images_from_csv(csv_path)
        else:
            if csv_path:
                print(f"提示: 未發現 CSV 檔案 '{csv_path}'，切換至自動篩選模式。")
            images = self.select_images_by_category(target_categories, min_count, max_images)
            
        # 執行下載
        self.download_images(images)

# ================= 測試執行區塊 =================
if __name__ == "__main__":
    # 設定參數
    CONFIG = {
        "json_path": "data/annotations_trainval2017/instances_train2017.json",
        "csv_path": "test.csv",
        "save_folder": "animal_images",
        "target_categories": ['dog', 'cat'],
        "min_animal_count": 2,
        "max_images": 100
    }

    try:
        selector = AnimalDataSelector(CONFIG["json_path"], CONFIG["save_folder"])
        
        selector.run(
            csv_path=CONFIG["csv_path"],
            target_categories=CONFIG["target_categories"],
            min_count=CONFIG["min_animal_count"],
            max_images=CONFIG["max_images"]
        )
    except Exception as e:
        print(f"發生錯誤: {e}")