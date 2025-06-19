import os
import shutil
import json # 新增導入 json 模組

def copy_file_to_immediate_subdirectories(source_file_path, target_parent_directory):
    """
    將指定的來源檔案複製到目標父資料夾下的所有直接子資料夾中 (僅一層)。

    Args:
        source_file_path (str): 要複製的來源檔案的路徑 (例如 "transforms.json")。
        target_parent_directory (str): 包含子資料夾的目標父資料夾的路徑。
    """
    # 1. 檢查來源檔案是否存在
    if not os.path.isfile(source_file_path):
        print(f"錯誤：來源檔案 '{source_file_path}' 不存在或不是一個檔案。")
        return

    # 2. 檢查目標父資料夾是否存在且是否為一個資料夾
    if not os.path.isdir(target_parent_directory):
        print(f"錯誤：目標資料夾 '{target_parent_directory}' 不存在或不是一個資料夾。")
        return

    source_filename = os.path.basename(source_file_path)
    copied_count = 0
    skipped_count = 0

    print(f"開始將檔案 '{source_filename}' 從 '{source_file_path}' 複製到 '{target_parent_directory}' 的下一層子資料夾中...")

    # 3. 遍歷目標父資料夾下的所有項目
    try:
        entries = os.listdir(target_parent_directory)
    except OSError as e:
        print(f"錯誤：無法讀取目標資料夾 '{target_parent_directory}' 的內容: {e}")
        return

    for entry_name in entries:
        potential_subdir_path = os.path.join(target_parent_directory, entry_name)
        
        # 檢查該項目是否為一個資料夾 (即下一層子資料夾)
        if os.path.isdir(potential_subdir_path):
            destination_file_path = os.path.join(potential_subdir_path, source_filename)
            try:
                # 執行複製操作，shutil.copy2 會同時複製檔案內容和元數據
                shutil.copy2(source_file_path, destination_file_path)
                print(f"  已成功複製到: {destination_file_path}")
                copied_count += 1
            except Exception as e:
                print(f"  複製到 '{destination_file_path}' 時發生錯誤: {e}")
                skipped_count += 1
            
    print(f"\n複製操作完成。")
    print(f"成功複製 {copied_count} 個檔案到下一層子資料夾。")
    if skipped_count > 0:
        print(f"跳過 {skipped_count} 個位置因發生錯誤。")

if __name__ == "__main__":
    # --- 請修改以下路徑 ---
    # 1. 指定來源 transforms.json 檔案的路徑
    source_transforms_file = "/home/cgvmis418/VideoGS/datasets/RUN_HiFi4G_location_9_30_T_PCD_DST_FT_process/transforms.json"  # 假設 transforms.json 在腳本執行的同一個目錄下

    # 2. 指定目標父資料夾的路徑
    target_directory = "/home/cgvmis418/VideoGS/datasets/RUN_HiFi4G_location_9_30_T_PCD_DST_FT_process"  # 假設目標資料夾結構在腳本執行的同一個目錄下
    # --- 路徑修改結束 ---

    print(f"準備從 '{source_transforms_file}' 複製檔案。")
    print(f"目標父資料夾為 '{target_directory}' (將複製到其下一層子資料夾)。")
    print("-" * 30)

    # 為了演示，如果 target_directory 不存在，我們可以創建一些範例資料夾結構
    if not os.path.exists(target_directory):
        print(f"提示：目標資料夾 '{target_directory}' 不存在，將為您創建一些範例資料夾結構以供測試。")
        try:
            os.makedirs(os.path.join(target_directory, "scene1", "sub_scene_A_deep"), exist_ok=True) # scene1 是下一層
            os.makedirs(os.path.join(target_directory, "scene2"), exist_ok=True) # scene2 是下一層
            os.makedirs(os.path.join(target_directory, "empty_scene_for_next_layer"), exist_ok=True) # empty_scene_for_next_layer 是下一層
            # 在目標父資料夾本身也創建一個檔案，以測試其不會被複製進去
            with open(os.path.join(target_directory, "some_file_in_parent.txt"), "w") as f:
                f.write("test parent")
            # 創建一個不在下一層的檔案，以測試其不會被複製進去
            with open(os.path.join(target_directory, "scene1", "sub_scene_A_deep", "deep_file.txt"), "w") as f:
                f.write("test deep")
            print(f"已創建範例資料夾於 '{target_directory}'。")
        except Exception as e:
            print(f"創建範例資料夾時出錯: {e}")
            exit()
    
    if not os.path.exists(source_transforms_file):
        print(f"提示：來源檔案 '{source_transforms_file}' 不存在，將為您創建一個範例檔案以供測試。")
        try:
            # 使用導入的 json 模組
            with open(source_transforms_file, "w") as f:
                json.dump({"description": "This is a sample transforms.json for next layer copy"}, f, indent=4)
            print(f"已創建範例來源檔案 '{source_transforms_file}'。")
        except Exception as e:
            print(f"創建範例來源檔案時出錯: {e}")
            exit()
            
    print("-" * 30)
    
    try:
        # 更新函數名稱
        copy_file_to_immediate_subdirectories(source_transforms_file, target_directory)
    except Exception as e:
        print(f"執行過程中發生未預期的錯誤: {e}")
        import traceback
        traceback.print_exc()
