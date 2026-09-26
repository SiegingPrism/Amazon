import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/code/business_entity_resolution"))

import polars as pl
import numpy as np
import joblib
from collections import defaultdict
from src.features import compute_features_precleaned
from src.utils import clean_text, evaluate_entity_f05
from test_new_pipeline import FastInvertedIndex

print("="*65)
print("TESTING INDIA PERFORMANCE BOOST WITH ENHANCED BLOCKING & DISAMBIGUATION")
print("="*65)

gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")

s1_india = s1_full.filter(pl.col("country") == "India")
india_eids = s1_india["entity_id"][:10000].to_list()
india_set = set(india_eids)

val_gt = gt_df.filter(pl.col("source1_entity_id").is_in(india_eids))
val_gt_dict = {}
needed_targets = set()
for r in val_gt.iter_rows(named=True):
    s1_id = r["source1_entity_id"]
    if r["matched_entity_ids"]:
        t = set(r["matched_entity_ids"].split(","))
        val_gt_dict[s1_id] = t
        needed_targets.update(t)
    else:
        val_gt_dict[s1_id] = set()

s1_val_dict = {r["entity_id"]: r for r in s1_india.filter(pl.col("entity_id").is_in(india_eids)).iter_rows(named=True)}

s2_full = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
s3_full = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")

s2_targets = {x for x in needed_targets if x.startswith("S2-")}
s3_targets = {x for x in needed_targets if x.startswith("S3-")}

s2_pool = pl.concat([s2_full.filter(pl.col("entity_id").is_in(list(s2_targets))), s2_full.filter(pl.col("country")=="India").sample(n=50000, seed=42)]).unique(subset=["entity_id"])
s3_pool = pl.concat([s3_full.filter(pl.col("entity_id").is_in(list(s3_targets))), s3_full.filter(pl.col("country")=="India").sample(n=50000, seed=42)]).unique(subset=["entity_id"])

target_records = {}
for r in s2_pool.iter_rows(named=True): target_records[r["entity_id"]] = r
for r in s3_pool.iter_rows(named=True): target_records[r["entity_id"]] = r

s1_clean = {r["entity_id"]: (clean_text(r["business_name"]), clean_text(r["business_address"]) if r["business_address"] else "") for r in s1_val_dict.values()}
target_clean = {r["entity_id"]: (clean_text(r["business_name"]), clean_text(r["business_address"]) if r["business_address"] else "", bool(r["business_address"])) for r in target_records.values()}

idx = FastInvertedIndex(max_key_len=5000)
eids = [r["entity_id"] for r in target_records.values()]
names = [r["business_name"] for r in target_records.values()]
addrs = [r["business_address"] for r in target_records.values()]
idx.add_records(eids, names, addrs)

total_gt = sum(len(v) for v in val_gt_dict.values())
hits = 0
val_pairs = []
val_feats = []

for s1_id in val_gt_dict:
    s1_r = s1_val_dict[s1_id]
    s1_cn, s1_ca = s1_clean[s1_id]
    c_indices = idx.query(s1_r["business_name"], s1_r["business_address"], top_name=25, top_addr=15)
    cand_eids = [idx.records[i][0] for i in c_indices]
    hits += len(val_gt_dict[s1_id] & set(cand_eids))
    for cid in cand_eids:
        if cid in target_clean:
            t_cn, t_ca, t_ha = target_clean[cid]
            val_feats.append(compute_features_precleaned(s1_cn, s1_ca, t_cn, t_ca, t_ha))
            val_pairs.append((s1_id, cid))

print(f"Enhanced India Candidate Recall: {hits}/{total_gt} ({hits/total_gt*100:.2f}%)")
print(f"Average Candidates per Entity:   {len(val_pairs)/len(val_gt_dict):.1f}")

model_artifact = joblib.load("code/business_entity_resolution/src/matching_model.joblib")
clf = model_artifact["model"]
probs = clf.predict_proba(np.array(val_feats, dtype=np.float32))[:, 1]

for th in [0.65, 0.70, 0.72, 0.74, 0.75, 0.78, 0.80, 0.82]:
    raw_matches = defaultdict(list)
    for (s1_id, cid), p in zip(val_pairs, probs):
        if p >= th:
            raw_matches[s1_id].append((cid, float(p)))
            
    # Bipartite Disambiguation
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
    precisions = []
    recalls = []
    for s1_id, gt_ids in val_gt_dict.items():
        pred_ids = final_preds.get(s1_id, set())
        scores.append(evaluate_entity_f05(gt_ids, pred_ids))
        if gt_ids:
            tp = len(gt_ids & pred_ids)
            p = tp / len(pred_ids) if pred_ids else 0.0
            r = tp / len(gt_ids)
            precisions.append(p)
            recalls.append(r)
            
    print(f"Th: {th:.2f} | India Macro F0.5: {np.mean(scores):.5f} | Precision: {np.mean(precisions):.5f} | Recall: {np.mean(recalls):.5f}")
