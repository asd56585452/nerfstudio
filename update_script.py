import json
import numpy as np
import os

def load_json(filepath):
    """載入 JSON 檔案"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    """儲存 JSON 檔案"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def convert_3x4_list_to_4x4_list(matrix_3x4_list):
    """將 3x4 的列表矩陣轉換為 4x4 的列表齊次矩陣"""
    if not (isinstance(matrix_3x4_list, list) and len(matrix_3x4_list) == 3 and
            all(isinstance(row, list) and len(row) == 4 for row in matrix_3x4_list)):
        raise ValueError("輸入必須是一個 3x4 的列表矩陣。")
    
    matrix_4x4_list = [row[:] for row in matrix_3x4_list] # 複製原始的 3x4 部分
    matrix_4x4_list.append([0.0, 0.0, 0.0, 1.0]) # 添加最後一行
    return matrix_4x4_list

def update_transforms_json(original_restored_path, target_transforms_path, output_path):
    """
    使用 original_restored.json 中的姿態更新 target_transforms.json，
    並將 applied_transform 設置為單位矩陣。

    Args:
        original_restored_path (str): 包含已還原姿態的 JSON 檔案路徑。
                                      (預期為列表，每個元素包含 'file_path' 和 3x4 'transform')
        target_transforms_path (str): 需要更新的 transforms.json 檔案路徑。
                                     (預期為字典，包含 'frames' 和 'applied_transform')
        output_path (str): 儲存更新後內容的 JSON 檔案路徑。
    """
    # 1. 載入 transforms_original_restored.json (還原後的姿態來源)
    # 這個檔案的頂層結構應該是一個列表
    restored_data_list = load_json(original_restored_path)
    if not isinstance(restored_data_list, list):
        print(f"錯誤：'{original_restored_path}' 的內容不是預期的列表格式。請檢查檔案。")
        return

    # 建立一個字典以便通過基本檔案名稱快速查找還原後的姿態
    restored_poses_map = {}
    for frame_info in restored_data_list:
        if "file_path" not in frame_info or "transform" not in frame_info:
            print(f"警告：在 '{original_restored_path}' 中找到格式不正確的影格資料：{frame_info}。已跳過。")
            continue
        base_filename = os.path.basename(frame_info["file_path"])
        restored_poses_map[base_filename] = frame_info["transform"] # 這是 3x4 的列表矩陣

    # 2. 載入目標 transforms.json 檔案
    target_data = load_json(target_transforms_path)
    if not isinstance(target_data, dict) or "frames" not in target_data:
        print(f"錯誤：'{target_transforms_path}' 的內容不是預期的字典格式或缺少 'frames' 鍵。請檢查檔案。")
        return

    # 3. 更新目標 transforms.json 中的影格
    updated_frames_count = 0
    not_found_frames_log = []
    
    frames_in_target = target_data.get("frames", [])
    if not isinstance(frames_in_target, list):
        print(f"錯誤：'{target_transforms_path}' 中的 'frames' 不是列表。請檢查檔案。")
        return

    for frame_in_target in frames_in_target:
        if "file_path" not in frame_in_target:
            print(f"警告：在 '{target_transforms_path}' 中找到缺少 'file_path' 的影格。已跳過。")
            continue
            
        target_base_filename = os.path.basename(frame_in_target["file_path"])
        
        if target_base_filename in restored_poses_map:
            # 從 restored_poses_map 中獲取 3x4 的姿態列表
            pose_3x4_list = restored_poses_map[target_base_filename]
            
            try:
                # 將 3x4 列表矩陣轉換為 4x4 列表矩陣
                pose_4x4_list = convert_3x4_list_to_4x4_list(pose_3x4_list)
                # 更新目標影格中的 transform_matrix
                frame_in_target["transform_matrix"] = pose_4x4_list
                updated_frames_count += 1
            except ValueError as e:
                print(f"警告：處理影格 '{target_base_filename}' 的姿態時出錯：{e}。已跳過此影格的更新。")
        else:
            not_found_frames_log.append(target_base_filename)

    if not_found_frames_log:
        print(f"警告：在 '{original_restored_path}' 中找不到以下 {len(not_found_frames_log)} 個影像的對應姿態：")
        for fname in not_found_frames_log:
            print(f"  - {fname}")

    # 4. 將 applied_transform 設置為 3x4 的單位矩陣
    # Nerfstudio 的 transforms.json 中的 applied_transform 通常是 3x4
    identity_3x4 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ]
    target_data["applied_transform"] = identity_3x4

    # 5. 儲存更新後的 target_data
    save_json(target_data, output_path)
    
    print(f"\n處理完成。")
    print(f"已成功更新 {updated_frames_count} 個影格的 'transform_matrix'。")
    print(f"已將 'applied_transform' 設置為 3x4 單位矩陣。")
    print(f"更新後的 transforms 檔案已儲存到: {output_path}")

if __name__ == "__main__":
    # --- 請修改以下路徑為您的實際檔案路徑 ---
    # 包含已還原姿態的 JSON 檔案 (來自上一個腳本的輸出)
    # 例如: "/path/to/your/transforms_original_restored.json"
    original_restored_file = "/home/cgvmis418/nerfstudio/outputs/0/splatfacto/20PCDALL:2025-07-08_134634/transforms_original_restored.json" 
                                                        
    # 您想要更新的 transforms.json 檔案
    # 例如: "/path/to/your/input_transforms.json"
    target_transforms_file = "/media/cgvmis418/新增磁碟區/2025-04-23_08-55-45/Ubnutu_RUN_HiFi4G_location_9_T_20PCD_30/nerfstudio_data_ALL/0/transforms.json" 
                                                
    # 更新後輸出的檔案名稱
    # 例如: "/path/to/your/output_transforms_updated.json"
    output_file = "/media/cgvmis418/新增磁碟區/2025-04-23_08-55-45/Ubnutu_RUN_HiFi4G_location_9_T_20PCD_30/nerfstudio_data_ALL/0/transforms.json"
    # --- 路徑修改結束 ---

    print(f"讀取已還原姿態來源檔案: {original_restored_file}")
    print(f"讀取目標 transforms 檔案: {target_transforms_file}")
    print(f"準備將結果寫入: {output_file}\n")

    # 執行更新操作
    # 確保您的 JSON 檔案與此腳本在同一目錄下，或者提供完整路徑
    try:
        if not os.path.exists(original_restored_file):
            print(f"錯誤：找不到來源檔案 '{original_restored_file}'。請檢查路徑。")
        elif not os.path.exists(target_transforms_file):
            print(f"錯誤：找不到目標檔案 '{target_transforms_file}'。請檢查路徑。")
        else:
            update_transforms_json(original_restored_file, target_transforms_file, output_file)
    except Exception as e:
        print(f"處理過程中發生未預期的錯誤: {e}")
        import traceback
        traceback.print_exc()
