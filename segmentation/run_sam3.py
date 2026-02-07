import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 1. 初始化模型
model = build_sam3_image_model(checkpoint_path="sam3/checkpoints/sam3.pt", load_from_HF=False)
processor = Sam3Processor(model)

# 2. 讀取影像與推論
image_pil = Image.open(r"assets\images\dog.jpg").convert("RGB")
inference_state = processor.set_image(image_pil)
prompt = "Outline the animal's silhouette and eyes separately."
output = processor.set_text_prompt(state=inference_state, prompt=prompt)

###################

print("---------- Output 內容大解密 ----------")
print(f"所有的 Keys: {output.keys()}")

# 1. 嘗試印出常見的 Label 欄位

if "labels" in output:
    print(f"Labels: {output['labels']}")
elif "phrases" in output:
    print(f"Phrases: {output['phrases']}")
elif "class_ids" in output:
    print(f"Class IDs: {output['class_ids']}")
else:
    print("⚠️ 輸出中沒有找到明顯的 Label 文字欄位。")
    print(f"因為你的提示詞只有 '{prompt}'，所以所有抓到的 Mask 預設都是這個標籤。")

# 2. 詳細列出每個偵測到的物件資訊
masks = output["masks"]
scores = output["scores"]
boxes = output["boxes"]

# 檢查是否有 pharases 或 labels 鍵值可供對應
detected_labels = None
if "phrases" in output:
    detected_labels = output["phrases"]
elif "labels" in output:
    detected_labels = output["labels"]

print(f"\n共偵測到 {len(scores)} 個物件：")
for i, score in enumerate(scores):
    # 取得這個物件的 Label (如果有)
    label_text = detected_labels[i] if detected_labels is not None else prompt
    
    # 取得 Box 座標
    box = boxes[i].cpu().numpy().astype(int)
    
    print(f"物件 #{i+1}:")
    print(f"  - 標籤 (Label): {label_text}")
    print(f"  - 分數 (Score): {score:.4f}")
    print(f"  - 位置 (Box)  : {box}")
    print("-" * 30)
###################

# 3. 準備繪圖 (PIL -> OpenCV BGR)
img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
overlay = img_cv.copy()

# 4. 繪製遮罩與框
if output["masks"] is not None:
    for mask, box in zip(output["masks"], output["boxes"]):
        # 隨機顏色
        color = np.random.randint(0, 255, 3).tolist()
        
        # 畫遮罩 (Mask)
        m = mask.cpu().numpy().squeeze() > 0
        overlay[m] = color
        
        # 畫框 (Box)
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)

    # 混合透明度 (Alpha Blending)
    cv2.addWeighted(overlay, 0.4, img_cv, 0.6, 0, img_cv)

# 5. 顯示結果
cv2.imshow("Result", img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()