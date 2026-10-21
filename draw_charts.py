import matplotlib.pyplot as plt
import numpy as np

# Cấu hình phong cách đồ họa
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Dữ liệu thực nghiệm B3
sigma_values = [0.4, 0.5, 0.6, 0.7, 0.8]
sigma_labels = ['σ = 0.4', 'σ = 0.5', 'σ = 0.6\n(Baseline)', 'σ = 0.7', 'σ = 0.8']

summe_f1 = [64.14, 65.57, 60.60, 67.25, 66.41]
tvsum_f1 = [62.25, 62.37, 62.58, 62.42, 62.61]

# ==========================================
# 1. BIỂU ĐỒ TẬP SUMME 
# ==========================================
fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

bars1 = ax1.bar(sigma_labels, summe_f1, color=['#a0c4ff', '#a0c4ff', '#ffadad', '#4895ef', '#a0c4ff'], width=0.55, zorder=2)
ax1.plot(sigma_labels, summe_f1, color='#1d3557', marker='o', linewidth=2, markersize=8, zorder=3)

# Điền giá trị số lên từng cột
for bar in bars1:
    yval = bar.get_height()
    fontweight = 'bold' if yval == max(summe_f1) else 'normal'
    color = '#0077b6' if yval == max(summe_f1) else '#333333'
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}%", ha='center', va='bottom', fontsize=10, fontweight=fontweight, color=color)

ax1.set_title('Ảnh hưởng của tham số $\sigma$ tới F1-Score trên tập SumMe', fontsize=13, fontweight='bold', pad=15)
ax1.set_xlabel('Giá trị ngưỡng Gaussian Kernel ($\sigma$)', fontsize=11, fontweight='bold', labelpad=10)
ax1.set_ylabel('Max F1-Score (%)', fontsize=11, fontweight='bold')
ax1.set_ylim(55, 71)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('chart_summe_sigma.png', dpi=300)
plt.close()
print("-> Đã lưu biểu đồ: chart_summe_sigma.png")

# ==========================================
# 2. BIỂU ĐỒ TẬP TVSUM (Avg F1-Score)
# ==========================================
fig, ax2 = plt.subplots(figsize=(8, 5), dpi=300)

bars2 = ax2.bar(sigma_labels, tvsum_f1, color=['#c8e6c9', '#c8e6c9', '#ffadad', '#c8e6c9', '#2e7d32'], width=0.55, zorder=2)
ax2.plot(sigma_labels, tvsum_f1, color='#1b4332', marker='s', linewidth=2, markersize=8, zorder=3)

# Điền giá trị số lên từng cột
for bar in bars2:
    yval = bar.get_height()
    fontweight = 'bold' if yval == max(tvsum_f1) else 'normal'
    color = '#1b4332' if yval == max(tvsum_f1) else '#333333'
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}%", ha='center', va='bottom', fontsize=10, fontweight=fontweight, color=color)

ax2.set_title('Ảnh hưởng của tham số $\sigma$ tới F1-Score trên tập TVSum', fontsize=13, fontweight='bold', pad=15)
ax2.set_xlabel('Giá trị ngưỡng Gaussian Kernel ($\sigma$)', fontsize=11, fontweight='bold', labelpad=10)
ax2.set_ylabel('Avg F1-Score (%)', fontsize=11, fontweight='bold')
ax2.set_ylim(60, 64)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('chart_tvsum_sigma.png', dpi=300)
plt.close()
print("-> Đã lưu biểu đồ: chart_tvsum_sigma.png")