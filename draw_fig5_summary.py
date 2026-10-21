import json
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from evaluation.generate_summary import generate_summary

def plot_video_summary_timeline(json_path, dataset_path, target_key, video_title, output_filename):
    json_files = [f for f in os.listdir(json_path) if f.endswith('.json')]
    json_files.sort(key=lambda x: int(x.split('.')[0][6:]) if x.split('.')[0][6:].isdigit() else -1)
    target_json = os.path.join(json_path, json_files[-1])
    
    with open(target_json, 'r') as f:
        data = json.load(f)
        
    init_scores = np.asarray(data[target_key])
    
    with h5py.File(dataset_path, 'r') as hdf:
        user_summary = np.array(hdf.get(target_key + '/user_summary'))
        sb = np.array(hdf.get(target_key + '/change_points'))
        n_frames = int(np.array(hdf.get(target_key + '/n_frames')))
        positions = np.array(hdf.get(target_key + '/picks'))

    # Sinh summary máy
    summaries = generate_summary([sb], [init_scores], [n_frames], [positions])
    model_summary = summaries[0]
    
    # Map init_scores ra toàn bộ n_frames
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    pos = positions.astype(int) if positions.dtype != int else positions
    if len(pos) == 0 or pos[-1] != n_frames:
        pos = np.concatenate([pos, [n_frames]])
    
    num_steps = min(len(init_scores), len(pos) - 1)
    for i in range(num_steps):
        p_left = min(int(pos[i]), n_frames)
        p_right = min(int(pos[i + 1]), n_frames)
        if p_left < p_right:
            frame_scores[p_left:p_right] = init_scores[i]

    # Ground truth
    gt_summary = user_summary.mean(axis=0) if user_summary.ndim > 1 else user_summary
    gt_binary = (gt_summary > 0.5).astype(int)

    # Vẽ biểu đồ 3 tầng
    fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True, gridspec_kw={'height_ratios': [2, 0.8, 0.8]})
    
    frames = np.arange(n_frames)
    
    # (a) Importance Score Curve
    axes[0].plot(frames, frame_scores, color='#1f77b4', linewidth=1.2, label='Importance Score (Predicted)')
    axes[0].set_ylabel('Score', fontsize=10)
    axes[0].set_title(f'Summary Timeline Visualization: {video_title} ({target_key})', fontsize=12, fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right')

    # (b) Model Selected Summary
    axes[1].fill_between(frames[:len(model_summary)], 0, model_summary, color='#2ca02c', alpha=0.8, label='Model Selected Summary')
    axes[1].set_yticks([])
    axes[1].set_ylabel('Model', fontsize=10, rotation=0, labelpad=25)
    axes[1].legend(loc='upper right')

    # (c) Ground Truth
    axes[2].fill_between(frames[:len(gt_binary)], 0, gt_binary, color='#ff7f0e', alpha=0.7, label='Ground Truth (User)')
    axes[2].set_yticks([])
    axes[2].set_xlabel('Frame Index', fontsize=10)
    axes[2].set_ylabel('GT', fontsize=10, rotation=0, labelpad=25)
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Đã xuất hình chuẩn đẹp: {output_filename}")

# Đường dẫn
summe_h5 = r'C:\Users\DELL\CA-SUM\model\data\SumMe\eccv16_dataset_summe_google_pool5.h5'
tvsum_h5 = r'C:\Users\DELL\CA-SUM\model\data\TVSum\eccv16_dataset_tvsum_google_pool5.h5'

path_summe_opt = r'C:\Users\DELL\CA-SUM\model\Summaries\exp1\reg0.7\SumMe\results\split0'
path_tvsum_opt = r'C:\Users\DELL\CA-SUM\model\Summaries\exp1\reg0.8\TVSum\results\split0'

# Chạy xuất ảnh
plot_video_summary_timeline(path_summe_opt, summe_h5, 'video_4', 'Air_Force_One (SumMe)', 'fig5_air_force_one.png')
plot_video_summary_timeline(path_tvsum_opt, tvsum_h5, 'video_14', 'Changing Memory Module (TVSum)', 'fig5_changing_ram.png')