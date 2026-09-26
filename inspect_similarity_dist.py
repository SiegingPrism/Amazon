import sys, os
sys.path.insert(0, os.path.abspath('code/business_entity_resolution'))

import polars as pl
from rapidfuzz import fuzz
from src.utils import clean_text

gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t").slice(0, 5000)
s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
s2_full = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
s3_full = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")

s1_dict = {r["entity_id"]: r for r in s1_full.filter(pl.col("entity_id").is_in(gt_df["source1_entity_id"].to_list())).iter_rows(named=True)}

needed_targets = set()
for r in gt_df.iter_rows(named=True):
    if r["matched_entity_ids"]:
        needed_targets.update(r["matched_entity_ids"].split(","))

target_records = {}
for r in s2_full.filter(pl.col("entity_id").is_in(list(needed_targets))).iter_rows(named=True):
    target_records[r["entity_id"]] = r
for r in s3_full.filter(pl.col("entity_id").is_in(list(needed_targets))).iter_rows(named=True):
    target_records[r["entity_id"]] = r

name_ratios = []
token_sets = []
addr_ratios = []

for r in gt_df.iter_rows(named=True):
    s1_id = r["source1_entity_id"]
    if not r["matched_entity_ids"]:
        continue
    r1 = s1_dict[s1_id]
    n1 = clean_text(r1["business_name"])
    a1 = clean_text(r1["business_address"])
    
    for tid in r["matched_entity_ids"].split(","):
        r2 = target_records.get(tid)
        if not r2: continue
        n2 = clean_text(r2["business_name"])
        a2 = clean_text(r2["business_address"])
        
        name_ratios.append(fuzz.ratio(n1, n2))
        token_sets.append(fuzz.token_set_ratio(n1, n2))
        if a2:
            addr_ratios.append(fuzz.token_set_ratio(a1, a2))

import numpy as np
print("True Matches Name fuzz.ratio percentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 99]:
    print(f"  P{p:02d}: {np.percentile(name_ratios, p):.1f}")

print("\nTrue Matches Name fuzz.token_set_ratio percentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 99]:
    print(f"  P{p:02d}: {np.percentile(token_sets, p):.1f}")

print(f"\nFraction with token_set_ratio < 40: {np.mean(np.array(token_sets) < 40):.4%}")
print(f"Fraction with token_set_ratio < 30: {np.mean(np.array(token_sets) < 30):.4%}")
