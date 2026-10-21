import subprocess
sigmas = [0.4, 0.5, 0.6, 0.7, 0.8]
block_sizes = [30, 60, 90, 120]
datasets = ["SumMe", "TVSum"]

print("==================================================")
print("   BẮT ĐẦU CHẠY THỰC NGHIỆM B3 CHO MÔ HÌNH CA-SUM  ")
print("==================================================")

# --- THỰC NGHIỆM 1: ĐỔI SIGMA (reg_factor) (Cố định block_size = 60) ---
print("\n[PHẦN 1] ĐÁNH GIÁ ẢNH HƯỞNG CỦA SIGMA (reg_factor)")
for ds in datasets:
    for sig in sigmas:
        cmd = f"python main.py --mode train --video_type {ds} --reg_factor {sig} --block_size 60"
        print(f"\n---> Running: Dataset={ds} | reg_factor={sig} | block_size=60")
        subprocess.run(cmd, shell=True)

# --- THỰC NGHIỆM 2: ĐỔI BLOCK_SIZE (M) (Cố định reg_factor = 0.6) ---
print("\n[PHẦN 2] ĐÁNH GIÁ ẢNH HƯỞNG CỦA BLOCK_SIZE (M)")
for ds in datasets:
    for bs in block_sizes:
        if bs == 60:
            continue 
        cmd = f"python main.py --mode train --video_type {ds} --reg_factor 0.6 --block_size {bs}"
        print(f"\n---> Running: Dataset={ds} | reg_factor=0.6 | block_size={bs}")
        subprocess.run(cmd, shell=True)

print("\n==================================================")
print("     ĐÃ HOÀN THÀNH TOÀN BỘ LƯỢT CHẠY THỰC NGHIỆM!  ")
print("==================================================")