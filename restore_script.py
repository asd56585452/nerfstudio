import json
import numpy as np

def load_json(filepath):
    """載入 JSON 檔案"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    """儲存 JSON 檔案"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def matrix_to_list(matrix):
    """將 NumPy 矩陣轉換為列表的列表"""
    return matrix.tolist()

def make_4x4_matrix(matrix_3x4):
    """將 3x4 矩陣擴展為 4x4 齊次矩陣"""
    matrix_4x4 = np.eye(4)
    matrix_4x4[:3, :] = np.array(matrix_3x4)
    return matrix_4x4

def restore_extrinsics(dataparser_transforms_path, transforms_train_path, output_path="transforms_original.json"):
    """
    還原原始相機外部參數。

    Args:
        dataparser_transforms_path (str): dataparser_transforms.json 的路徑。
        transforms_train_path (str): transforms_train.json 的路徑 (包含已轉換的相機參數)。
        output_path (str): 儲存還原後參數的 JSON 檔案路徑。
    """
    # 1. 載入 dataparser_transforms.json
    dp_data = load_json(dataparser_transforms_path)
    T_dp_3x4 = np.array(dp_data["transform"])
    s_dp = dp_data["scale"]

    # 將 dataparser 的 transform 擴展為 4x4 並計算其逆矩陣
    T_dp_4x4 = np.eye(4)
    T_dp_4x4[:3, :] = T_dp_3x4
    try:
        inv_T_dp_4x4 = np.linalg.inv(T_dp_4x4)
    except np.linalg.LinAlgError:
        print("錯誤：dataparser_transforms.json 中的轉換矩陣是奇異的，無法計算逆矩陣。")
        return

    # 2. 載入 transforms_train.json
    train_data = load_json(transforms_train_path)
    
    restored_frames = []

    # 3. 遍歷 transforms_train.json 中的每一個 frame
    for frame_data in train_data:
        P_ns_3x4 = np.array(frame_data["transform"]) # Nerfstudio 使用的相機參數 (已轉換)
        
        # 將 Nerfstudio 的相機參數擴展為 4x4
        P_ns_4x4 = make_4x4_matrix(P_ns_3x4)

        # 創建中間轉換矩陣 P_intermediate
        # 首先撤銷縮放操作 (僅對平移部分)
        P_intermediate_4x4 = np.copy(P_ns_4x4)
        P_intermediate_4x4[:3, 3] = P_ns_4x4[:3, 3] / s_dp
        
        # 撤銷 dataparser 的全局轉換
        # P_original = inv(T_dp) @ P_intermediate
        P_original_4x4 = inv_T_dp_4x4 @ P_intermediate_4x4
        
        # 提取還原後的 3x4 外部參數矩陣
        P_original_3x4 = P_original_4x4[:3, :]
        
        # 建立新的 frame 資料
        restored_frame = {
            "file_path": frame_data.get("file_path", ""), # 保留原始 file_path
            "transform": matrix_to_list(P_original_3x4)
        }
        # 如果原始 frame 中有其他欄位，也可以考慮保留
        for key, value in frame_data.items():
            if key not in ["transform", "file_path"]:
                restored_frame[key] = value
        
        restored_frames.append(restored_frame)

    # 4. 儲存還原後的參數
    save_json(restored_frames, output_path)
    print(f"還原後的相機參數已儲存到: {output_path}")

if __name__ == "__main__":
    # 請將以下路徑替換為您檔案的實際路徑
    dataparser_transforms_file = "/home/cgvmis418/nerfstudio/outputs/0/splatfacto/2025-05-30_183304/dataparser_transforms.json"
    transforms_train_file = "/home/cgvmis418/nerfstudio/outputs/0/splatfacto/2025-05-30_183304/transforms_train.json"
    output_file = "/home/cgvmis418/nerfstudio/outputs/0/splatfacto/2025-05-30_183304/transforms_original_restored.json"

    # 執行還原操作
    # 確保您的 JSON 檔案與此腳本在同一目錄下，或者提供完整路徑
    try:
        restore_extrinsics(dataparser_transforms_file, transforms_train_file, output_file)
    except FileNotFoundError:
        print(f"錯誤：找不到輸入檔案。請確保 '{dataparser_transforms_file}' 和 '{transforms_train_file}' 存在於正確的路徑。")
    except Exception as e:
        print(f"還原過程中發生錯誤: {e}")

