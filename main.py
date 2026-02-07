import os
import configparser

# 蒐集資料到資料夾 ( 可選測資指定圖片 or COCO類別指定 )
from data_selector import AnimalDataSelector
from animal_extractor import AnimalDetector
from eyes_extractor import AnimalEyePipeline
from measurement_tool import AnimalMeasurementTool


# =========================================================
# Config Loader
# =========================================================
def load_config(config_path: str = "config.ini") -> dict:
    """Load INI config file and convert to typed dict."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 找不到設定檔: {config_path}")

    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")

    return {
        "data_prep": {
            "coco_json": parser["data_prep"]["coco_json"],
            "test_csv": parser["data_prep"]["test_csv"],
            "target_categories": [
                x.strip() for x in parser["data_prep"]["target_categories"].split(",")
            ],
            "min_animal_count": parser.getint("data_prep", "min_animal_count"),
            "max_download": parser.getint("data_prep", "max_download"),
        },
        "paths": {
            "image_folder": parser["paths"]["image_folder"],
            "json_file": parser["paths"]["json_file"],
            "output_visual_folder": parser["paths"]["output_visual_folder"],
        },
        "models": {
            "yolo_ckpt": parser["models"]["yolo_ckpt"],
            "sam3_ckpt": parser["models"]["sam3_ckpt"],
        },
        "flags": {
            "show_visualization": parser.getboolean("flags", "show_visualization"),
        },
    }


# =========================================================
# Pipeline Stages
# =========================================================
def stage_0_data_preparation(cfg: dict):
    print("\n" + "=" * 40)
    print("Stage 0: Data Preparation")
    print("=" * 40)

    selector = AnimalDataSelector(
        json_path=cfg["data_prep"]["coco_json"],
        save_folder=cfg["paths"]["image_folder"],
    )

    selector.run(
        csv_path=cfg["data_prep"]["test_csv"],
        target_categories=cfg["data_prep"]["target_categories"],
        min_count=cfg["data_prep"]["min_animal_count"],
        max_images=cfg["data_prep"]["max_download"],
    )


def stage_1_yolo_segmentation(cfg: dict):
    print("\n" + "=" * 40)
    print("Stage 1: YOLO Segmentation")
    print("=" * 40)

    image_folder = cfg["paths"]["image_folder"]
    if not os.path.exists(image_folder) or not os.listdir(image_folder):
        raise RuntimeError("❌ 圖片資料夾為空，請確認 Stage 0 是否成功")

    detector = AnimalDetector(
        input_folder=image_folder,
        output_json=cfg["paths"]["json_file"],
        model_path=cfg["models"]["yolo_ckpt"],
    )
    detector.run()


def stage_2_eye_detection(cfg: dict):
    print("\n" + "=" * 40)
    print("Stage 2: SAM3 Eye Detection")
    print("=" * 40)

    json_file = cfg["paths"]["json_file"]
    if not os.path.exists(json_file):
        raise RuntimeError("❌ 找不到 YOLO 輸出 JSON，請確認 Stage 1")

    pipeline = AnimalEyePipeline(
        json_path=json_file,
        image_root=cfg["paths"]["image_folder"],
        sam3_ckpt=cfg["models"]["sam3_ckpt"],
        output_json=None,  # in-place update
        show=cfg["flags"]["show_visualization"],
    )
    pipeline.run()


def stage_3_measurement(cfg: dict):
    print("\n" + "=" * 40)
    print("Stage 3: Measurement")
    print("=" * 40)

    os.makedirs(cfg["paths"]["output_visual_folder"], exist_ok=True)

    measurer = AnimalMeasurementTool(
        json_file_path=cfg["paths"]["json_file"],
        image_folder_path=cfg["paths"]["image_folder"],
        output_folder=cfg["paths"]["output_visual_folder"],
        show=cfg["flags"]["show_visualization"],
    )
    measurer.run()


# =========================================================
# Main
# =========================================================
def main():
    try:
        cfg = load_config()
    except Exception as e:
        print(f"設定檔載入失敗: {e}")
        return

    stage_0_data_preparation(cfg)
    stage_1_yolo_segmentation(cfg)
    stage_2_eye_detection(cfg)
    stage_3_measurement(cfg)

    print("=" * 40)
    print(f"[輸入] Input: {cfg['paths']['image_folder']}")
    print(f"[輸出] Img Ouput: {cfg['paths']['output_visual_folder']} ") 
    print(f"[輸出] Json Ouput: {cfg['paths']['json_file']}")


if __name__ == "__main__":
    main()
