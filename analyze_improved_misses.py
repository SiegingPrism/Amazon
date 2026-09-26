import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/code/business_entity_resolution"))

import polars as pl
from collections import defaultdict
from test_new_pipeline import FastInvertedIndex, extract_improved_keys

gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")

val_slice = gt_df.slice(25000, 10000)
val_gt_dict = {}
needed_targets = set()
for r in val_slice.iter_rows(named=True):
    s1_id = r["source1_entity_id"]
    if r["matched_entity_ids"]:
        t = set(r["matched_entity_ids"].split(","))
        val_gt_dict[s1_id] = t
        needed_targets.update(t)
    else:
        val_gt_dict[s1_id] = set()
        
s1_val_df = s1_full.filter(pl.col("entity_id").is_in(list(val_gt_dict.keys())))
s1_val_dict = {r["entity_id"]: r for r in s1_val_df.iter_rows(named=True)}

s2_full = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
s3_full = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")

s2_targets = {x for x in needed_targets if x.startswith("S2-")}
s3_targets = {x for x in needed_targets if x.startswith("S3-")}

s2_pool = pl.concat([s2_full.filter(pl.col("entity_id").is_in(list(s2_targets))), s2_full.sample(n=100000, seed=42)]).unique(subset=["entity_id"])
s3_pool = pl.concat([s3_full.filter(pl.col("entity_id").is_in(list(s3_targets))), s3_full.sample(n=100000, seed=42)]).unique(subset=["entity_id"])

target_records = {}
for r in s2_pool.iter_rows(named=True): target_records[r["entity_id"]] = r
for r in s3_pool.iter_rows(named=True): target_records[r["entity_id"]] = r

indices = {"US": FastInvertedIndex(max_key_len=3000), "India": FastInvertedIndex(max_key_len=3000)}
recs_by_c = defaultdict(lambda: ([], [], []))
for tid, tr in target_records.items():
    c = tr["country"]
    if c in indices:
        recs_by_c[c][0].append(tr["entity_id"])
        recs_by_c[c][1].append(tr["business_name"])
        recs_by_c[c][2].append(tr["business_address"])
        
for c in indices:
    eids, names, addrs = recs_by_c[c]
    indices[c].add_records(eids, names, addrs)
    
missed = []
for s1_id, gt_ids in val_gt_dict.items():
    if not gt_ids: continue
    s1_r = s1_val_dict[s1_id]
    c = s1_r["country"]
    if c in indices:
        c_indices = indices[c].query(s1_r["business_name"], s1_r["business_address"], top_name=18, top_addr=12)
        cands = set(indices[c].records[i][0] for i in c_indices)
        for gid in gt_ids:
            if gid not in cands:
                tr = target_records.get(gid, {})
                missed.append((s1_r, tr))

print(f"Total Missed Target Pairs: {len(missed)} / {sum(len(v) for v in val_gt_dict.values())}")
print("\nSample Missed Target Pairs:")
for s1_r, tr in missed[:15]:
    print(f"S1: [{s1_r.get('country')}] {s1_r.get('business_name')} | {s1_r.get('business_address')}")
    print(f"TR: [{tr.get('country')}] {tr.get('business_name')} | {tr.get('business_address')}")
    s1_k = set(extract_improved_keys(s1_r.get('business_name'), s1_r.get('business_address')))
    tr_k = set(extract_improved_keys(tr.get('business_name'), tr.get('business_address')))
    overlap = s1_k & tr_k
    print(f"    S1 keys: {len(s1_k)}, TR keys: {len(tr_k)}, Common keys: {overlap}\n")
