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

LEET_MAP = {
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s",
    "6": "g", "7": "t", "8": "b"
}

def deleet_token(tok):
    if re.search(r"[a-zA-Z]", tok) and re.search(r"[0-9]", tok):
        return "".join(LEET_MAP.get(c, c) for c in tok)
    return tok

def deleet_text(text):
    if not text: return ""
    tokens = text.split()
    return " ".join(deleet_token(t) for t in tokens)

print("="*75)
print("TESTING LEET RECOVERY & FP PRUNING ON 10,000 VALIDATION ENTITIES")
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

# Clean both standard and deleeted
s1_clean = {r["entity_id"]: (clean_text(deleet_text(r["business_name"])), clean_text(r["business_address"]) if r["business_address"] else "") for r in s1_dict.values()}
target_clean = {r["entity_id"]: (clean_text(deleet_text(r["business_name"])), clean_text(r["business_address"]) if r["business_address"] else "", bool(r["business_address"])) for r in target_records.values()}

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

model_artifact = joblib.load("code/business_entity_resolution/src/matching_model.joblib")
clf = model_artifact["model"]

val_pairs = []
val_feats = []
val_countries = []
val_n_sims = []
val_num_matches = []

for s1_id in gt_dict:
    s1_r = s1_dict[s1_id]
    c = s1_r["country"]
    s1_cn, s1_ca = s1_clean[s1_id]
    if c in indices:
        c_indices = indices[c].query(deleet_text(s1_r["business_name"]), s1_r["business_address"], top_name=16, top_addr=8)
        cand_eids = [indices[c].records[i][0] for i in c_indices]
        for cid in cand_eids:
            if cid in target_clean:
                t_cn, t_ca, t_ha = target_clean[cid]
                feats = compute_features_precleaned(s1_cn, s1_ca, t_cn, t_ca, t_ha)
                val_feats.append(feats)
                val_pairs.append((s1_id, cid))
                val_countries.append(c)
                val_n_sims.append(feats[2]) # n_token_set
                val_num_matches.append(feats[10]) # num_match

probs = clf.predict_proba(np.array(val_feats, dtype=np.float32))[:, 1]

# Baseline evaluation (without extra guard)
COUNTRY_THRESHOLDS = {
    "US": {"addr": 0.80, "no_addr": 0.84},
    "India": {"addr": 0.82, "no_addr": 0.78}
}

for apply_fp_guard in [False, True]:
    raw_matches = defaultdict(list)
    for (s1_id, cid), p, c, n_sim, num_m, feat in zip(val_pairs, probs, val_countries, val_n_sims, val_num_matches, val_feats):
        has_addr = feat[6] == 0.0 # addr_null is feat[6]
        th = COUNTRY_THRESHOLDS[c]["addr"] if has_addr else COUNTRY_THRESHOLDS[c]["no_addr"]
        
        if apply_fp_guard:
            # Guard against multi-tenant false merges:
            # 1. Conflicting numbers (num_m == 0.0) with low name similarity (n_sim < 0.45)
            # 2. Complete absence of name similarity (n_sim < 0.20) unless address is near-perfect (a_token_set > 0.95 and num_m == 1.0)
            if num_m == 0.0 and n_sim < 0.45:
                continue
            if n_sim < 0.20 and (num_m != 1.0 or feat[7] < 0.95):
                continue
                
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

    scores = []
    fps = 0
    fns = 0
    tps = 0
    for s1_id in gt_dict:
        gt = gt_dict[s1_id]
        pred = final_preds.get(s1_id, set())
        sc = evaluate_entity_f05(gt, pred)
        scores.append(sc)
        fps += len(pred - gt)
        fns += len(gt - pred)
        tps += len(gt & pred)

    prec = tps / (tps + fps) if (tps + fps) > 0 else 0
    rec = tps / (tps + fns) if (tps + fns) > 0 else 0
    guard_label = "WITH FP Guard + Deleet" if apply_fp_guard else "Baseline Dual-Threshold + Deleet"
    print(f"\n--- {guard_label} ---")
    print(f"Macro F0.5:  {np.mean(scores):.6f}")
    print(f"Precision:   {prec*100:.2f}% (TPs={tps}, FPs={fps})")
    print(f"Recall:      {rec*100:.2f}% (FNs={fns})")
