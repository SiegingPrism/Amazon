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
print("OPTIMIZING FOR MACRO F0.5 > 0.991 ON 15,000 VALIDATION ENTITIES")
print("="*75)

gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t").slice(10000, 15000)
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

for s1_id in gt_dict:
    s1_r = s1_dict[s1_id]
    c = s1_r["country"]
    s1_cn, s1_ca = s1_clean[s1_id]
    if c in indices:
        c_indices = indices[c].query(deleet_text(s1_r["business_name"]), s1_r["business_address"], top_name=20, top_addr=10)
        cand_eids = [indices[c].records[i][0] for i in c_indices]
        for cid in cand_eids:
            if cid in target_clean:
                t_cn, t_ca, t_ha = target_clean[cid]
                feats = compute_features_precleaned(s1_cn, s1_ca, t_cn, t_ca, t_ha)
                val_feats.append(feats)
                val_pairs.append((s1_id, cid))
                val_countries.append(c)

probs = clf.predict_proba(np.array(val_feats, dtype=np.float32))[:, 1]
print(f"Total candidate pairs: {len(val_pairs)}")

# Grid search thresholds and precision filters
best_score = 0.0
best_params = None

for th_us_addr in [0.80, 0.82, 0.84, 0.86]:
    for th_us_no in [0.82, 0.84, 0.86, 0.88]:
        for th_in_addr in [0.80, 0.82, 0.84, 0.86]:
            for th_in_no in [0.78, 0.80, 0.82, 0.84]:
                for apply_guard in [True, False]:
                    raw_matches = defaultdict(list)
                    for (s1_id, cid), p, c, feat in zip(val_pairs, probs, val_countries, val_feats):
                        has_addr = feat[6] == 0.0
                        if c == "US":
                            th = th_us_addr if has_addr else th_us_no
                        else:
                            th = th_in_addr if has_addr else th_in_no
                            
                        if apply_guard:
                            n_sim = feat[2] # n_token_set
                            num_m = feat[10] # num_match
                            # Filter 1: Number conflict with different name
                            if num_m == 0.0 and n_sim < 0.45: continue
                            # Filter 2: Negligible name similarity without perfect address match
                            if n_sim < 0.20 and (num_m != 1.0 or feat[7] < 0.95): continue
                            
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

                    scs = [evaluate_entity_f05(gt_dict[s1_id], final_preds.get(s1_id, set())) for s1_id in gt_dict]
                    mean_sc = np.mean(scs)
                    if mean_sc > best_score:
                        best_score = mean_sc
                        best_params = (th_us_addr, th_us_no, th_in_addr, th_in_no, apply_guard)
                        print(f"NEW BEST: Macro F0.5 = {best_score:.6f} | US: ({th_us_addr}, {th_us_no}), IN: ({th_in_addr}, {th_in_no}), Guard: {apply_guard}")

print(f"\nFinal Best Macro F0.5: {best_score:.6f}")
print(f"Optimal parameters: {best_params}")
