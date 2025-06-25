import os
from PIL import Image
from pathlib import Path

# --- 1. 設定路徑 ---
# 將 'path/to/project/output' 替換成你 ns-process-data 的輸出資料夾路徑
project_path = Path("/media/cgvmis418/新增磁碟區/2025-05-25_17-44-00/GRADF_HiFi4G_location_9_T_PCD/nerfstudio_data/0") 

# --- 不需要修改以下部分 ---
image_dir = project_path / "images"
mask_dir = project_path / "masks"

# 檢查圖片資料夾是否存在
if not image_dir.exists():
    print(f"錯誤：找不到圖片資料夾 '{image_dir}'")
    exit()

# 創建 masks 資料夾
mask_dir.mkdir(exist_ok=True)
print(f"將從 '{image_dir}' 創建遮罩到 '{mask_dir}'...")

# 遍歷所有圖片
image_files = sorted(list(image_dir.iterdir()))
for image_file in image_files:
    if image_file.suffix.lower() not in ['.png']:
        print(f"跳過非 PNG 檔案: {image_file.name}")
        continue

    try:
        with Image.open(image_file) as img:
            # 檢查圖片是否有 Alpha 通道
            if img.mode == 'RGBA':
                # 提取 Alpha 通道
                alpha_channel = img.split()[-1]
                
                # 建立遮罩檔案的儲存路徑
                mask_save_path = mask_dir / image_file.name
                
                # 儲存 Alpha 通道為新的圖片
                alpha_channel.save(mask_save_path)
                print(f"已為 '{image_file.name}' 生成遮罩")
            else:
                print(f"警告：圖片 '{image_file.name}' 沒有 Alpha 通道，無法生成遮罩。")

    except Exception as e:
        print(f"處理檔案 '{image_file.name}' 時發生錯誤: {e}")

print("\n遮罩生成完畢！")