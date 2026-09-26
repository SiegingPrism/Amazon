import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/code/business_entity_resolution"))

import polars as pl
import numpy as np
import joblib, re
from collections import defaultdict
from rapidfuzz import fuzz
from src.features import compute_features_precleaned
from src.utils import clean_text, evaluate_entity_f05, clean_number
from test_ultimate_keys import UltimateInvertedIndex

print("="*75)
print("LARGE-SCALE VALIDATION EVALUATION (25,000 S1 ENTITIES)")
print("="*75)

# Load 25,000 GT entities
gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t").slice(10000, 25000)
s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
s2_full = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
s3_full = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")

gt_dict = {r["source1_entity_id"]: set(r["matched_entity_ids"].split(",")) if r["matched_entity_ids"] else set() for r in gt_df.iter_rows(named=True)}
needed = set.union(*gt_dict.values())

s1_dict = {r["entity_id"]: r for r in s1_full.filter(pl.col("entity_id").is_in(list(gt_dict.keys()))).iter_rows(named=True)}
s2_pool = pl.concat([s2_full.filter(pl.col("entity_id").is_in(list(needed))), s2_full.sample(n=150000, seed=42)]).unique(subset=["entity_id"])
s3_pool = pl.concat([s3_full.filter(pl.col("entity_id").is_in(list(needed))), s3_full.sample(n=150000, seed=42)]).unique(subset=["entity_id"])

target_records = {}
for r in s2_pool.iter_rows(named=True): target_records[r["entity_id"]] = r
for r in s3_pool.iter_rows(named=True): target_records[r["entity_id"]] = r

s1_clean = {r["entity_id"]: (clean_text(r["business_name"]), clean_text(r["business_address"]) if r["business_address"] else "") for r in s1_dict.values()}
target_clean = {r["entity_id"]: (clean_text(r["business_name"]), clean_text(r["business_address"]) if r["business_address"] else "", bool(r["business_address"])) for r in target_records.values()}

indices = {"US": UltimateInvertedIndex(max_key_len=1500), "India": UltimateInvertedIndex(max_key_len=1500)}
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

model_artifact = joblib.load("code/business_entity_resolution/src/matching_model.joblib")
clf = model_artifact["model"]

val_pairs = []
val_feats = []
val_countries = []

for s1_id in gt_dict:
    s1_r = s1_dict[s1_id]
    c = s1_r["country"]
    s1_cn, s1_ca = s1_clean[s1_id]
    if c in indices:
        c_indices = indices[c].query(s1_r["business_name"], s1_r["business_address"], top_name=16, top_addr=8)
        cand_eids = [indices[c].records[i][0] for i in c_indices]
        for cid in cand_eids:
            if cid in target_clean:
                t_cn, t_ca, t_ha = target_clean[cid]
                feats = compute_features_precleaned(s1_cn, s1_ca, t_cn, t_ca, t_ha)
                val_feats.append(feats)
                val_pairs.append((s1_id, cid))
                val_countries.append(c)

probs = clf.predict_proba(np.array(val_feats, dtype=np.float32))[:, 1]
print(f"Total candidate pairs evaluated: {len(val_pairs)}")

# Evaluate current production pipeline
raw_matches = defaultdict(list)
for (s1_id, cid), p, c, feat in zip(val_pairs, probs, val_countries, val_feats):
    has_addr = feat[6] == 0.0
    if c == "India":
        th = 0.82 if has_addr else 0.78
    else:
        th = 0.80 if has_addr else 0.84
    if p >= th:
        raw_matches[s1_id].append((cid, float(p)))

target_winner = {}
for s1_id, m_list in raw_matches.items():
    for cid, p in m_list:
        if cid not in target_winner or p > target_winner[cid][1]:
            target_winner[cid] = (s1_id, p)

final_preds = defaultdict(set)
for s1_id, m_list in raw_matches.items():
    for cid, p in m_list:
        if target_winner.get(cid, (None,))[0] == s1_id:
            final_preds[s1_id].add(cid)

c_scores = defaultdict(list)
for s1_id in gt_dict:
    gt = gt_dict[s1_id]
    pred = final_preds.get(s1_id, set())
    sc = evaluate_entity_f05(gt, pred)
    c = s1_dict[s1_id]["country"]
    c_scores[c].append(sc)

print("\n--- Current Pipeline Performance ---")
all_sc = []
for c, scs in c_scores.items():
    print(f"[{c}] Macro F0.5 ({len(scs)} entities): {np.mean(scs):.6f}")
    all_sc.extend(scs)
print(f"Overall Validation Macro F0.5: {np.mean(all_sc):.6f}")
