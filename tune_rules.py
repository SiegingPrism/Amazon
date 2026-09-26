import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/code/business_entity_resolution"))

import polars as pl
import numpy as np
import joblib
from collections import defaultdict
from rapidfuzz import fuzz
from src.features import compute_features_precleaned
from src.utils import clean_text, evaluate_entity_f05, LEGAL_STOPWORDS, ADDR_STOPWORDS
from test_ultimate_keys import UltimateInvertedIndex

print("="*75)
print("GRID SEARCH & TUNING: PRECISION GUARDRAIL & RECALL RECOVERY")
print("="*75)

# Load 10k validation set
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
val_meta = [] # (cn1, ca1, cn2, ca2, ha2)

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
                val_meta.append((s1_cn, s1_ca, t_cn, t_ca, t_ha))

probs = clf.predict_proba(np.array(val_feats, dtype=np.float32))[:, 1]
print(f"Scored {len(val_pairs):,} validation pairs.")

def evaluate_predictions(final_pred_map):
    scores = []
    us_scores = []
    in_scores = []
    for s1_id, gt in gt_dict.items():
        pr = final_pred_map.get(s1_id, set())
        sc = evaluate_entity_f05(gt, pr)
        scores.append(sc)
        c = s1_dict[s1_id]["country"]
        if c == "US": us_scores.append(sc)
        elif c == "India": in_scores.append(sc)
    # Also add France synthetic (100% singletons = 1.0)
    # France is 14.975% of test, US is 38.274%, India is 46.751%
    us_m = np.mean(us_scores)
    in_m = np.mean(in_scores)
    overall_macro = 0.14975 * 1.0 + 0.38274 * us_m + 0.46751 * in_m
    return np.mean(scores), us_m, in_m, overall_macro

# Precompute name similarity metrics for fast grid search
meta_metrics = []
for cn1, ca1, cn2, ca2, ha2 in val_meta:
    # Token set ratio
    tsr = fuzz.token_set_ratio(cn1, cn2)
    # Ratio
    fr = fuzz.ratio(cn1, cn2)
    # Word overlap
    w1 = set(w for w in cn1.split() if w not in LEGAL_STOPWORDS and len(w) >= 3)
    w2 = set(w for w in cn2.split() if w not in LEGAL_STOPWORDS and len(w) >= 3)
    has_common_word = bool(w1 & w2)
    meta_metrics.append((tsr, fr, has_common_word, ha2, ca1, ca2))

print("Precomputed metrics.")

# 1. Test Baseline
base_matches = defaultdict(list)
for (s1_id, cid), p, c in zip(val_pairs, probs, val_countries):
    th = 0.82 if c == "India" else 0.79
    if p >= th:
        base_matches[s1_id].append((cid, float(p)))

tw = {}
for s1_id, ml in base_matches.items():
    for cid, p in ml:
        if cid not in tw or p > tw[cid][1]:
            tw[cid] = (s1_id, p)
base_final = defaultdict(set)
for s1_id, ml in base_matches.items():
    for cid, p in ml:
        if tw.get(cid, (None,))[0] == s1_id:
            base_final[s1_id].add(cid)

val_m, us_m, in_m, test_est = evaluate_predictions(base_final)
print(f"BASELINE: Val_F0.5={val_m:.5f} | US={us_m:.5f} | IN={in_m:.5f} | Estimated Test F0.5={test_est:.5f}")

# 2. Grid Search over Guardrails & Recovery
best_test_est = test_est
best_config = None

# Guardrail: If address matches but name similarity is too low, reject
# name_floor: minimum token_set_ratio required when not sharing words
for name_floor in [30, 35, 40, 45, 50, 55]:
    for null_addr_th in [0.20, 0.35, 0.50, 0.65, 0.80]:
        for null_addr_min_sim in [80, 85, 90, 95]:
            cand_matches = defaultdict(list)
            for i in range(len(val_pairs)):
                s1_id, cid = val_pairs[i]
                p = probs[i]
                c = val_countries[i]
                tsr, fr, has_common_word, ha2, ca1, ca2 = meta_metrics[i]
                
                # Rule 1: Precision Guardrail
                # If neither has_common_word nor high similarity, reject!
                if not has_common_word and tsr < name_floor and fr < name_floor:
                    continue
                    
                # Base country threshold
                c_th = 0.82 if c == "India" else 0.79
                
                # Rule 2: Missing Address Recall Recovery
                # If target has NO address, but name is virtually identical, lower threshold
                effective_th = c_th
                if not ha2 and tsr >= null_addr_min_sim:
                    effective_th = min(c_th, null_addr_th)
                    
                if p >= effective_th:
                    cand_matches[s1_id].append((cid, float(p)))
                    
            # Disambiguation
            tw = {}
            for s1_id, ml in cand_matches.items():
                for cid, p in ml:
                    if cid not in tw or p > tw[cid][1]:
                        tw[cid] = (s1_id, p)
            final_p = defaultdict(set)
            for s1_id, ml in cand_matches.items():
                for cid, p in ml:
                    if tw.get(cid, (None,))[0] == s1_id:
                        final_p[s1_id].add(cid)
                        
            val_m, us_m, in_m, test_est = evaluate_predictions(final_p)
            if test_est > best_test_est:
                best_test_est = test_est
                best_config = (name_floor, null_addr_th, null_addr_min_sim)
                print(f"--> NEW BEST: Test F0.5={test_est:.5f} (+{(test_est - 0.98604)*10000:.1f} bps) | Val={val_m:.5f} | US={us_m:.5f} | IN={in_m:.5f} | name_floor={name_floor}, null_th={null_addr_th}, null_sim={null_addr_min_sim}")

print("\n" + "="*75)
print(f"OPTIMAL CONFIG: {best_config}")
print(f"BEST ESTIMATED TEST F0.5: {best_test_est:.6f}")
print("="*75)
