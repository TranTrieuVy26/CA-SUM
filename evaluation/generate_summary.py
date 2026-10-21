import numpy as np


def solve_knapsack_greedy(max_len, lengths, scores):
    """Thuật toán Greedy Knapsack giải quyết bài toán chọn Key-shot an toàn tuyệt đối."""
    n = len(lengths)
    if n == 0 or max_len <= 0:
        return []

    # Tính tỷ lệ điểm / độ dài
    ratios = [(scores[i] / max(lengths[i], 1), i) for i in range(n)]
    # Sắp xếp giảm dần theo tỷ lệ
    ratios.sort(key=lambda x: x[0], reverse=True)

    selected = []
    current_len = 0
    for ratio, idx in ratios:
        if current_len + lengths[idx] <= max_len:
            selected.append(idx)
            current_len += lengths[idx]

    return selected


def generate_summary(all_shot_bound, all_scores, all_nframes, all_positions):
    """Generate automatic machine summary."""
    all_summaries = []
    for video_index in range(len(all_scores)):
        shot_bound = all_shot_bound[video_index]
        frame_init_scores = all_scores[video_index]
        n_frames = all_nframes[video_index]
        positions = all_positions[video_index]

        frame_scores = np.zeros(n_frames, dtype=np.float32)
        if positions.dtype != int:
            positions = positions.astype(np.int32)
        if len(positions) == 0 or positions[-1] != n_frames:
            positions = np.concatenate([positions, [n_frames]])

        num_steps = min(len(frame_init_scores), len(positions) - 1)
        for i in range(num_steps):
            pos_left = min(int(positions[i]), n_frames)
            pos_right = min(int(positions[i + 1]), n_frames)
            if pos_left < pos_right:
                frame_scores[pos_left:pos_right] = frame_init_scores[i]

        shot_imp_scores = []
        shot_lengths = []
        valid_shot_bound = []

        for shot in shot_bound:
            shot_start = min(int(shot[0]), n_frames - 1)
            shot_end = min(int(shot[1]), n_frames - 1)
            length = shot_end - shot_start + 1

            if length > 0 and shot_start <= shot_end:
                shot_lengths.append(length)
                shot_imp_scores.append(float(frame_scores[shot_start:shot_end + 1].mean()))
                valid_shot_bound.append([shot_start, shot_end])

        if len(valid_shot_bound) == 0:
            all_summaries.append(np.zeros(n_frames, dtype=np.int8))
            continue

        final_shot = valid_shot_bound[-1]
        max_frame_idx = max(int(final_shot[1]), n_frames - 1)
        
        # ĐÃ CĂN CHUẨN THỤT LỀ TẠI ĐÂY (8 SPACES)
        # final_max_length = int((max_frame_idx + 1) * 0.10)
        final_max_length = int((max_frame_idx + 1) * 0.20)

        selected = solve_knapsack_greedy(final_max_length, shot_lengths, shot_imp_scores)

        summary = np.zeros(max_frame_idx + 1, dtype=np.int8)
        for shot_idx in selected:
            s_start = valid_shot_bound[shot_idx][0]
            s_end = valid_shot_bound[shot_idx][1]
            summary[s_start:s_end + 1] = 1

        all_summaries.append(summary)

    return all_summaries