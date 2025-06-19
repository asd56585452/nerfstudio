import json
import os

def load_json(filepath):
    """載入 JSON 檔案"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    """儲存 JSON 檔案"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def update_video_transforms_from_source(source_transforms_path, target_videoGS_path, output_path):
    """
    使用 source_transforms.json 中的姿態 (基於 colmap_im_id) 更新 target_videoGS_transforms.json。

    Args:
        source_transforms_path (str): 包含來源姿態和 'colmap_im_id' 的 JSON 檔案路徑。
                                      (預期為字典，包含 'frames')
        target_videoGS_path (str): 需要更新的 videoGS_transforms.json 檔案路徑。
                                   (預期為字典，包含 'frames')
        output_path (str): 儲存更新後內容的 JSON 檔案路徑。
    """
    # 1. 載入來源 transforms.json (姿態來源)
    source_data = load_json(source_transforms_path)
    if not isinstance(source_data, dict) or "frames" not in source_data:
        print(f"錯誤：來源檔案 '{source_transforms_path}' 的內容不是預期的字典格式或缺少 'frames' 鍵。請檢查檔案。")
        return

    # 建立一個字典以便通過 {colmap_im_id}.png 快速查找來源姿態
    source_poses_map = {}
    for frame_info in source_data.get("frames", []):
        if "colmap_im_id" not in frame_info or "transform_matrix" not in frame_info:
            print(f"警告：在來源檔案 '{source_transforms_path}' 中找到格式不正確的影格資料：{frame_info}。已跳過。")
            continue
        
        # 確保 transform_matrix 是 4x4
        source_transform_matrix = frame_info["transform_matrix"]
        if not (isinstance(source_transform_matrix, list) and len(source_transform_matrix) == 4 and
                all(isinstance(row, list) and len(row) == 4 for row in source_transform_matrix)):
            print(f"警告：來源檔案 '{source_transforms_path}' 中影格 colmap_im_id {frame_info['colmap_im_id']} 的 'transform_matrix' 不是 4x4 列表。已跳過。")
            continue

        # 使用 colmap_im_id 構造檔案名稱鍵，例如 "1.png", "2.png"
        key_filename = f"{frame_info['colmap_im_id']}.png"
        source_poses_map[key_filename] = source_transform_matrix

    if not source_poses_map:
        print(f"錯誤：未能從來源檔案 '{source_transforms_path}' 載入任何有效的姿態。請檢查檔案內容和格式。")
        return

    # 2. 載入目標 videoGS_transforms.json 檔案
    target_data = load_json(target_videoGS_path)
    if not isinstance(target_data, dict) or "frames" not in target_data:
        print(f"錯誤：目標檔案 '{target_videoGS_path}' 的內容不是預期的字典格式或缺少 'frames' 鍵。請檢查檔案。")
        return

    # 3. 更新目標 videoGS_transforms.json 中的影格
    updated_frames_count = 0
    not_found_frames_log = []
    
    target_frames = target_data.get("frames", [])
    if not isinstance(target_frames, list):
        print(f"錯誤：目標檔案 '{target_videoGS_path}' 中的 'frames' 不是列表。請檢查檔案。")
        return

    for frame_in_target in target_frames:
        if "file_path" not in frame_in_target:
            print(f"警告：在目標檔案 '{target_videoGS_path}' 中找到缺少 'file_path' 的影格。已跳過。")
            continue
            
        # 從目標檔案的 file_path 中獲取基本檔案名稱 (例如 "1.png")
        target_base_filename = os.path.basename(frame_in_target["file_path"])
        
        if target_base_filename in source_poses_map:
            # 從 source_poses_map 中獲取 4x4 的姿態列表
            source_pose_4x4_list = source_poses_map[target_base_filename]
            
            # 更新目標影格中的 transform_matrix
            frame_in_target["transform_matrix"] = source_pose_4x4_list
            updated_frames_count += 1
        else:
            not_found_frames_log.append(target_base_filename)

    if not_found_frames_log:
        print(f"警告：在來源檔案 '{source_transforms_path}' (使用 colmap_im_id 映射) 中找不到以下 {len(not_found_frames_log)} 個目標影像的對應姿態：")
        for fname in not_found_frames_log:
            print(f"  - {fname}")

    # 4. 儲存更新後的 target_data
    save_json(target_data, output_path)
    
    print(f"\n處理完成。")
    print(f"已成功更新 {updated_frames_count} 個影格的 'transform_matrix'。")
    print(f"更新後的 videoGS transforms 檔案已儲存到: {output_path}")

if __name__ == "__main__":
    # --- 請修改以下路徑為您的實際檔案路徑 ---
    # 包含來源姿態和 'colmap_im_id' 的 JSON 檔案
    # 例如: "/path/to/your/source_transforms_with_colmap_id.json"
    source_transforms_file = "/media/cgvmis418/新增磁碟區/2025-04-23_08-55-45/RUN_HiFi4G_location_9_30_T_PCD/nerfstudio_data/0/transforms.json" 
                                                        
    # 您想要更新的 videoGS_transforms.json 檔案
    # 例如: "/path/to/your/input_videoGS_transforms.json"
    target_videoGS_file = "/home/cgvmis418/VideoGS/datasets/RUN_HiFi4G_location_9_30_T_PCD_DST_FT_process/transforms.json" 
                                                
    # 更新後輸出的檔案名稱
    # 例如: "/path/to/your/output_videoGS_transforms_updated.json"
    output_file = "/home/cgvmis418/VideoGS/datasets/RUN_HiFi4G_location_9_30_T_PCD_DST_FT_process/transforms.json"
    # --- 路徑修改結束 ---

    print(f"讀取來源姿態檔案 (含 colmap_im_id): {source_transforms_file}")
    print(f"讀取目標 videoGS_transforms 檔案: {target_videoGS_file}")
    print(f"準備將結果寫入: {output_file}\n")

    # 執行更新操作
    try:
        if not os.path.exists(source_transforms_file):
            print(f"錯誤：找不到來源檔案 '{source_transforms_file}'。請檢查路徑。")
        elif not os.path.exists(target_videoGS_file):
            print(f"錯誤：找不到目標檔案 '{target_videoGS_file}'。請檢查路徑。")
        else:
            update_video_transforms_from_source(source_transforms_file, target_videoGS_file, output_file)
    except Exception as e:
        print(f"處理過程中發生未預期的錯誤: {e}")
        import traceback
        traceback.print_exc()
