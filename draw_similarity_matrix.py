import h5py
import numpy as np
import matplotlib.pyplot as plt

def compute_gaussian_similarity(features, sigma):
    """Tính ma trận tương đồng Gaussian S(i,j) = exp(-||f(i) - f(j)||^2 / (2 * sigma^2))"""
    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    norm_features = features / norms

    # Tính khoảng cách Euclidean bình phương giữa mọi cặp vector
    # ||f(i) - f(j)||^2 = ||f(i)||^2 + ||f(j)||^2 - 2 * f(i) . f(j)
    dot_product = np.dot(norm_features, norm_features.T)
    sq_distances = 2.0 - 2.0 * dot_product
    sq_distances = np.maximum(sq_distances, 0) # Tránh số âm nhỏ do sai số làm tròn

    # Gaussian Kernel
    similarity_matrix = np.exp(-sq_distances / (2.0 * (sigma ** 2)))
    return similarity_matrix

# Đọc đặc trưng video từ file H5 (Dùng video_14 của TVSum hoặc video_4 của SumMe)
tvsum_h5 = r'C:\Users\DELL\CA-SUM\model\data\TVSum\eccv16_dataset_tvsum_google_pool5.h5'

with h5py.File(tvsum_h5, 'r') as hdf:
    features = np.array(hdf.get('video_14/features'))

# Tính ma trận cho sigma = 0.4 và sigma = 0.8
S_sigma_04 = compute_gaussian_similarity(features, sigma=0.4)
S_sigma_08 = compute_gaussian_similarity(features, sigma=0.8)

# Vẽ biểu đồ so sánh 2 ma trận
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

im1 = axes[0].imshow(S_sigma_04, cmap='viridis', aspect='auto', origin='upper')
axes[0].set_title(r'Similarity Matrix ($\sigma = 0.4$)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Frame Index $j$', fontsize=10)
axes[0].set_ylabel('Frame Index $i$', fontsize=10)
fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

im2 = axes[1].imshow(S_sigma_08, cmap='viridis', aspect='auto', origin='upper')
axes[1].set_title(r'Similarity Matrix ($\sigma = 0.8$)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Frame Index $j$', fontsize=10)
axes[1].set_ylabel('Frame Index $i$', fontsize=10)
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

plt.suptitle('Gaussian Kernel Similarity Matrix Comparison on Video 14', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('fig6_similarity_matrix_comparison.png', dpi=300)
plt.close()

print("Đã xuất hình thành công: fig6_similarity_matrix_comparison.png")