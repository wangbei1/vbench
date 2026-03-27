"""
Read VBench_full_info.json and split prompts into per-dimension txt files.
Output files are saved to a 'prompts_by_dimension/' folder next to this script.
"""
import json
import os
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
info_path = os.path.join(script_dir, "VBench_full_info.json")
output_dir = os.path.join(script_dir, "prompts_by_dimension")

os.makedirs(output_dir, exist_ok=True)

with open(info_path, "r", encoding="utf-8") as f:
    data = json.load(f)

dim_prompts = defaultdict(list)
for item in data:
    prompt = item["prompt_en"]
    for dim in item["dimension"]:
        dim_prompts[dim].append(prompt)

for dim, prompts in sorted(dim_prompts.items()):
    out_path = os.path.join(output_dir, f"{dim}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p + "\n")
    print(f"{dim}: {len(prompts)} prompts -> {dim}.txt")
