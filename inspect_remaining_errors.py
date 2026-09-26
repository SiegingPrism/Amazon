import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/code/business_entity_resolution"))

import polars as pl
import numpy as np
import joblib
from collections import defaultdict
from src.features import compute_features_precleaned
from src.utils import clean_text, evaluate_entity_f05
from test_ultimate_keys import UltimateInvertedIndex

print("="*75)
print("DEEP DIVE: EXACT RESIDUAL ERRORS IN US & INDIA")
print("="*75)

gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t").slice(25000, 10000)
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

s1_clean = {r["entity_id"]: (clean_text(r["business_name"]), clean_text(r["business_address"]) if r["business_address"] else "") for r in s1_dict.values()}
target_clean = {r["entity_id"]: (clean_text(r["business_name"]), clean_text(r["business_address"]) if r["business_address"] else "", bool(r["business_address"])) for r in target_records.values()}

indices = {"US": UltimateInvertedIndex(max_key_len=5000), "India": UltimateInvertedIndex(max_key_len=5000)}
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
        c_indices = indices[c].query(s1_r["business_name"], s1_r["business_address"], top_name=25, top_addr=15)
        cand_eids = [indices[c].records[i][0] for i in c_indices]
        for cid in cand_eids:
            if cid in target_clean:
                t_cn, t_ca, t_ha = target_clean[cid]
                val_feats.append(compute_features_precleaned(s1_cn, s1_ca, t_cn, t_ca, t_ha))
                val_pairs.append((s1_id, cid))
                val_countries.append(c)

probs = clf.predict_proba(np.array(val_feats, dtype=np.float32))[:, 1]

# Apply our calibrated thresholds: US=0.79, India=0.82
raw_matches = defaultdict(list)
for (s1_id, cid), p, c in zip(val_pairs, probs, val_countries):
    th = 0.82 if c == "India" else 0.79
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

# Inspect errors by country
for country in ["US", "India"]:
    c_s1 = [sid for sid, r in s1_dict.items() if r["country"] == country]
    c_fps = []
    c_fns = []
    c_scores = []
    for s1_id in c_s1:
        gt = gt_dict[s1_id]
        pred = final_preds.get(s1_id, set())
        sc = evaluate_entity_f05(gt, pred)
        c_scores.append(sc)
        s1_r = s1_dict[s1_id]
        for cid in (pred - gt):
            tr = target_records.get(cid, {})
            c_fps.append((s1_r, tr))
        for cid in (gt - pred):
            tr = target_records.get(cid, {})
            c_fns.append((s1_r, tr))
            
    print(f"\n[{country}] Macro F0.5: {np.mean(c_scores):.5f} | Total FPs: {len(c_fps)} | Total FNs: {len(c_fns)}")
    print(f"Sample FPs for {country}:")
    for s1_r, tr in c_fps[:6]:
        print(f"  FP: S1: {s1_r.get('business_name')} | {s1_r.get('business_address')}")
        print(f"      TR: {tr.get('business_name')} | {tr.get('business_address')}")
    print(f"\nSample FNs for {country}:")
    for s1_r, tr in c_fns[:6]:
        print(f"  FN: S1: {s1_r.get('business_name')} | {s1_r.get('business_address')}")
        print(f"      TR: {tr.get('business_name')} | {tr.get('business_address')}")
