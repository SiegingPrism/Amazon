import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/code/business_entity_resolution"))

import polars as pl
import numpy as np
from collections import defaultdict
from rapidfuzz import fuzz
from src.features import compute_features_precleaned
from src.utils import clean_text, evaluate_entity_f05, clean_number
from test_ultimate_keys import UltimateInvertedIndex
import re

LEET_MAP = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b"}
def deleet_token(tok):
    if re.search(r"[a-zA-Z]", tok) and re.search(r"[0-9]", tok):
        return "".join(LEET_MAP.get(c, c) for c in tok)
    return tok
def deleet_text(text):
    if not text: return ""
    return " ".join(deleet_token(t) for t in text.split())

print("="*75)
print("AUDITING EVERY REMAINING ERROR IN 10,000 VALIDATION ENTITIES")
print("="*75)

gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t").slice(10000, 10000)
s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
s2_full = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
s3_full = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")

gt_dict = {r["source1_entity_id"]: set(r["matched_entity_ids"].split(",")) if r["matched_entity_ids"] else set() for r in gt_df.iter_rows(named=True)}
needed = set.union(*gt_dict.values())

s1_dict = {r["entity_id"]: r for r in s1_full.filter(pl.col("entity_id").is_in(list(gt_dict.keys()))).iter_rows(named=True)}
s2_pool = pl.concat([s2_full.filter(pl.col("entity_id").is_in(list(needed))), s2_full.sample(n=100000, seed=42)]).unique(subset=["entity_id"])
s3_pool = pl.concat([s3_full.filter(pl.col("entity_id").is_in(list(needed))), s3_full.sample(n=100000, seed=42)]).unique(subset=["entity_id"])

target_records = {}
for r in s2_pool.iter_rows(named=True): target_records[r["entity_id"]] = r
for r in s3_pool.iter_rows(named=True): target_records[r["entity_id"]] = r

# Analyze ground truth pairs that NEVER get retrieved by blocking
indices = {"US": UltimateInvertedIndex(max_key_len=1500), "India": UltimateInvertedIndex(max_key_len=1500)}
recs_by_c = defaultdict(lambda: ([], [], []))
for tid, tr in target_records.items():
    c = tr["country"]
    if c in indices:
        recs_by_c[c][0].append(tr["entity_id"])
        recs_by_c[c][1].append(deleet_text(tr["business_name"]))
        recs_by_c[c][2].append(tr["business_address"])

for c in indices:
    eids, names, addrs = recs_by_c[c]
    indices[c].add_records(eids, names, addrs)

unretrieved_gt = []
retrieved_gt_count = 0
total_gt_pairs = 0

for s1_id, gt in gt_dict.items():
    s1_r = s1_dict[s1_id]
    c = s1_r["country"]
    total_gt_pairs += len(gt)
    if c in indices:
        c_indices = indices[c].query(deleet_text(s1_r["business_name"]), s1_r["business_address"], top_name=25, top_addr=15)
        cand_eids = set(indices[c].records[i][0] for i in c_indices)
        retrieved_gt_count += len(gt & cand_eids)
        for tid in (gt - cand_eids):
            unretrieved_gt.append((s1_r, target_records.get(tid, {})))

print(f"Total GT pairs: {total_gt_pairs}")
print(f"Retrieved by blocking: {retrieved_gt_count} ({retrieved_gt_count/total_gt_pairs*100:.2f}%)")
print(f"Missed by blocking: {len(unretrieved_gt)} ({len(unretrieved_gt)/total_gt_pairs*100:.2f}%)")

print("\n--- SAMPLE GT PAIRS MISSED BY BLOCKING (Why did blocking fail?) ---")
for s1_r, tr in unretrieved_gt[:15]:
    print(f"Country: {s1_r.get('country')}")
    print(f"  S1: {s1_r.get('business_name')} | {s1_r.get('business_address')}")
    print(f"  TR: {tr.get('business_name')} | {tr.get('business_address')}")
    print("-" * 60)
