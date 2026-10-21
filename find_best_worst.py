import json
import numpy as np
import h5py
import os
from evaluation.evaluation_metrics import evaluate_summary
from evaluation.generate_summary import generate_summary

def get_all_video_fscores(json_path, dataset_path, eval_method):
    json_files = [f for f in os.listdir(json_path) if f.endswith('.json')]
    json_files.sort(key=lambda x: int(x.split('.')[0][6:]) if x.split('.')[0][6:].isdigit() else -1)
    target_json = os.path.join(json_path, json_files[-1])
    
    with open(target_json, 'r') as f:
        data = json.load(f)
        
    results = {}
    with h5py.File(dataset_path, 'r') as hdf:
        for key in data.keys():
            scores = np.asarray(data[key])
            user_summary = np.array(hdf.get(key + '/user_summary'))
            sb = np.array(hdf.get(key + '/change_points'))
            n_frames = np.array(hdf.get(key + '/n_frames'))
            positions = np.array(hdf.get(key + '/picks'))

            summaries = generate_summary([sb], [scores], [n_frames], [positions])
            f_score = evaluate_summary(summaries[0], user_summary, eval_method)
            results[key] = f_score
            
    return results

# SumMe
summe_h5 = r'C:\Users\DELL\CA-SUM\model\data\SumMe\eccv16_dataset_summe_google_pool5.h5'
summe_base_path = r'C:\Users\DELL\CA-SUM\model\Summaries\exp1\reg0.6\SumMe\results\split0'
summe_opt_path = r'C:\Users\DELL\CA-SUM\model\Summaries\exp1\reg0.7\SumMe\results\split0'

summe_base = get_all_video_fscores(summe_base_path, summe_h5, 'max')
summe_opt = get_all_video_fscores(summe_opt_path, summe_h5, 'max')

# TVSum
tvsum_h5 = r'C:\Users\DELL\CA-SUM\model\data\TVSum\eccv16_dataset_tvsum_google_pool5.h5'
tvsum_base_path = r'C:\Users\DELL\CA-SUM\model\Summaries\exp1\reg0.6\TVSum\results\split0'
tvsum_opt_path = r'C:\Users\DELL\CA-SUM\model\Summaries\exp1\reg0.8\TVSum\results\split0'

tvsum_base = get_all_video_fscores(tvsum_base_path, tvsum_h5, 'avg')
tvsum_opt = get_all_video_fscores(tvsum_opt_path, tvsum_h5, 'avg')

print("=== SUMME RESULTS (σ=0.7) ===")
for k in summe_opt:
    print(f"{k}: Baseline={summe_base[k]:.2f}%, Optimal={summe_opt[k]:.2f}%")

print("\n=== TVSUM RESULTS (σ=0.8) ===")
for k in tvsum_opt:
    print(f"{k}: Baseline={tvsum_base[k]:.2f}%, Optimal={tvsum_opt[k]:.2f}%")