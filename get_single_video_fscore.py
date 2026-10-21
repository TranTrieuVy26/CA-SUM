import json
import numpy as np
import h5py
import os
from evaluation.evaluation_metrics import evaluate_summary
from evaluation.generate_summary import generate_summary

def get_video_fscore(json_path, dataset_path, target_key, eval_method):
    json_files = [f for f in os.listdir(json_path) if f.endswith('.json')]
    json_files.sort(key=lambda x: int(x.split('.')[0][6:]) if x.split('.')[0][6:].isdigit() else -1)
    
    # Lấy json ở epoch tối ưu (hoặc epoch có F1 max)
    target_json = os.path.join(json_path, json_files[-1])
    
    with open(target_json, 'r') as f:
        data = json.load(f)
        
    if target_key not in data:
        print(f"Không tìm thấy {target_key} trong file {target_json}")
        return None

    scores = np.asarray(data[target_key])
    
    with h5py.File(dataset_path, 'r') as hdf:
        user_summary = np.array(hdf.get(target_key + '/user_summary'))
        sb = np.array(hdf.get(target_key + '/change_points'))
        n_frames = np.array(hdf.get(target_key + '/n_frames'))
        positions = np.array(hdf.get(target_key + '/picks'))

    summaries = generate_summary([sb], [scores], [n_frames], [positions])
    gen_summary = summaries[0]
    
    f_score = evaluate_summary(gen_summary, user_summary, eval_method)
    return f_score

# --- ĐƯỜNG DẪN H5 ---
summe_h5 = r'C:\Users\DELL\CA-SUM\model\data\SumMe\eccv16_dataset_summe_google_pool5.h5'
tvsum_h5 = r'C:\Users\DELL\CA-SUM\model\data\TVSum\eccv16_dataset_tvsum_google_pool5.h5'

# --- ĐƯỜNG DẪN KẾT QUẢ ---
path_summe_base = r'C:\Users\DELL\CA-SUM\model\Summaries\exp1\reg0.6\SumMe\results\split0'
path_summe_opt = r'C:\Users\DELL\CA-SUM\model\Summaries\exp1\reg0.7\SumMe\results\split0'

path_tvsum_base = r'C:\Users\DELL\CA-SUM\model\Summaries\exp1\reg0.6\TVSum\results\split0'
path_tvsum_opt = r'C:\Users\DELL\CA-SUM\model\Summaries\exp1\reg0.8\TVSum\results\split0'

# 1. Air_Force_One (video_4)
f1_air_base = get_video_fscore(path_summe_base, summe_h5, 'video_4', 'max')
f1_air_opt = get_video_fscore(path_summe_opt, summe_h5, 'video_4', 'max')

# 2. Changing Memory Module (video_14)
f1_ram_base = get_video_fscore(path_tvsum_base, tvsum_h5, 'video_14', 'avg')
f1_ram_opt = get_video_fscore(path_tvsum_opt, tvsum_h5, 'video_14', 'avg')

print("="*55)
print("KẾT QUẢ THỰC NGHIỆM CHI TIẾT TRÊN TỪNG VIDEO tiêu biểu:")
print("="*55)
print(f"Video Air_Force_One (video_4 - SumMe):")
print(f"  - Baseline (σ=0.6) F1 : {f1_air_base:.2f}%")
print(f"  - Optimal  (σ=0.7) F1 : {f1_air_opt:.2f}%")
print(f"  - Chênh lệch ΔF1       : +{f1_air_opt - f1_air_base:.2f}%")
print("-" * 55)
print(f"Video Changing Memory Module (video_14 - TVSum):")
print(f"  - Baseline (σ=0.6) F1 : {f1_ram_base:.2f}%")
print(f"  - Optimal  (σ=0.8) F1 : {f1_ram_opt:.2f}%")
print(f"  - Chênh lệch ΔF1       : +{f1_ram_opt - f1_ram_base:.2f}%")
print("="*55)