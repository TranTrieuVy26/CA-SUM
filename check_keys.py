import json
import os

path_summe = r'C:\Users\DELL\CA-SUM\model\Summaries\exp1\reg0.7\SumMe\results\split0'
json_files = [f for f in os.listdir(path_summe) if f.endswith('.json')]
json_files.sort(key=lambda x: int(x.split('.')[0][6:]) if x.split('.')[0][6:].isdigit() else -1)

with open(os.path.join(path_summe, json_files[-1]), 'r') as f:
    data = json.load(f)

print("=== CÁC VIDEO CÓ TRONG TẬP SUMME ===")
for k in data.keys():
    print(k)

path_tvsum = r'C:\Users\DELL\CA-SUM\model\Summaries\exp1\reg0.8\TVSum\results\split0'
json_files_tv = [f for f in os.listdir(path_tvsum) if f.endswith('.json')]
json_files_tv.sort(key=lambda x: int(x.split('.')[0][6:]) if x.split('.')[0][6:].isdigit() else -1)

with open(os.path.join(path_tvsum, json_files_tv[-1]), 'r') as f:
    data_tv = json.load(f)

print("\n=== CÁC VIDEO CÓ TRONG TẬP TVSUM ===")
for k in data_tv.keys():
    print(k)