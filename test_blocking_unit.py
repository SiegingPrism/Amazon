import sys, os
sys.path.insert(0, os.path.abspath('code/business_entity_resolution'))

import polars as pl
import time
from src.blocking import InvertedIndex
from src.utils import evaluate_macro_f05

print("Testing InvertedIndex blocking...")
# Load small sample of 1,000 S1 from train
gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t").slice(0, 1000)
s1_needed = set(gt_df["source1_entity_id"].to_list())
targets = set()
gt_dict = {}
for r in gt_df.iter_rows(named=True):
    s1_id = r["source1_entity_id"]
    if r["matched_entity_ids"]:
        t = set(r["matched_entity_ids"].split(","))
        gt_dict[s1_id] = t
        targets.update(t)
    else:
        gt_dict[s1_id] = set()

s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(s1_needed)))
s2_df = pl.read_csv("dataset/train/train_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(targets)))
s3_df = pl.read_csv("dataset/train/train_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(targets)))

idx = InvertedIndex(max_key_len=1000)
idx.add_records(s2_df["entity_id"].to_list(), s2_df["business_name"].to_list(), s2_df["business_address"].to_list())
idx.add_records(s3_df["entity_id"].to_list(), s3_df["business_name"].to_list(), s3_df["business_address"].to_list())

found = 0
total = sum(len(t) for t in gt_dict.values())
cand_counts = []

for r in s1_df.iter_rows(named=True):
    top_cands = idx.query(r["business_name"], r["business_address"], top_k=20)
    cand_ids = {idx.records[i][0] for i in top_cands}
    cand_counts.append(len(cand_ids))
    t = gt_dict[r["entity_id"]]
    found += len(cand_ids & t)

print(f"Candidate Recall on 1,000 entities: {found} / {total} = {found/total:.2%}")
print(f"Avg candidates per entity: {sum(cand_counts)/len(cand_counts):.1f}")
