import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/code/business_entity_resolution"))

import polars as pl
import numpy as np
import joblib
from collections import defaultdict
from src.features import compute_features_precleaned
from src.utils import clean_text, evaluate_entity_f05
from src.blocking import InvertedIndex, extract_blocking_keys

def diagnose_india():
    print("="*60)
    print("DIAGNOSING INDIA ENTITY RESOLUTION PERFORMANCE")
    print("="*60)
    
    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
    s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
    
    # Filter for India entities only
    s1_india = s1_full.filter(pl.col("country") == "India")
    india_ids = set(s1_india["entity_id"][:10000])
    
    val_gt = gt_df.filter(pl.col("source1_entity_id").is_in(list(india_ids)))
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
            
    s1_val_dict = {r["entity_id"]: r for r in s1_india.filter(pl.col("entity_id").is_in(list(val_gt_dict.keys()))).iter_rows(named=True)}
    
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
    
    idx = InvertedIndex(max_key_len=1000)
    for r in target_records.values():
        idx.records.append((r["entity_id"], r["business_name"], r["business_address"]))
        idx_pos = len(idx.records) - 1
        for k in set(extract_blocking_keys(r["business_name"], r["business_address"])):
            idx.index[k].append(idx_pos)
            
    model_artifact = joblib.load("code/business_entity_resolution/src/matching_model.joblib")
    clf = model_artifact["model"]
    
    total_gt = sum(len(v) for v in val_gt_dict.values())
    hits = 0
    fn_not_retrieved = 0
    fn_low_prob = 0
    fp_count = 0
    
    missed_blocking = []
    low_prob_matches = []
    
    for s1_id, gt_ids in val_gt_dict.items():
        s1_r = s1_val_dict[s1_id]
        s1_cn, s1_ca = s1_clean[s1_id]
        c_indices = idx.query(s1_r["business_name"], s1_r["business_address"], top_k=20)
        cand_eids = set(idx.records[i][0] for i in c_indices)
        
        retrieved_gt = gt_ids & cand_eids
        hits += len(retrieved_gt)
        for gid in (gt_ids - cand_eids):
            fn_not_retrieved += 1
            if len(missed_blocking) < 10:
                tr = target_records.get(gid, {})
                missed_blocking.append((s1_r, tr))
                
        cand_feats = [compute_features_precleaned(s1_cn, s1_ca, *target_clean[cid]) for cid in cand_eids if cid in target_clean]
        if cand_feats:
            cand_list = [cid for cid in cand_eids if cid in target_clean]
            probs = clf.predict_proba(np.array(cand_feats, dtype=np.float32))[:, 1]
            for cid, p in zip(cand_list, probs):
                if cid in gt_ids and p < 0.80:
                    fn_low_prob += 1
                    if len(low_prob_matches) < 10:
                        tr = target_records[cid]
                        low_prob_matches.append((s1_r, tr, p))
                        
    print(f"Total True India Matches: {total_gt}")
    print(f"Candidate Recall:          {hits}/{total_gt} ({hits/total_gt*100:.2f}%)")
    print(f"Missed in Blocking:        {fn_not_retrieved} ({fn_not_retrieved/total_gt*100:.2f}%)")
    print(f"Low Probability (< 0.80):  {fn_low_prob} ({fn_low_prob/total_gt*100:.2f}%)")
    
    print("\nSAMPLE MISSED IN BLOCKING:")
    for s1_r, tr in missed_blocking[:5]:
        print(f"  S1: {s1_r['business_name']} | {s1_r['business_address']}")
        print(f"  TR: {tr.get('business_name')} | {tr.get('business_address')}\n")
        
    print("\nSAMPLE LOW PROBABILITY MATCHES:")
    for s1_r, tr, p in low_prob_matches[:5]:
        print(f"  S1: {s1_r['business_name']} | {s1_r['business_address']}")
        print(f"  TR: {tr.get('business_name')} | {tr.get('business_address')} (p={p:.4f})\n")

if __name__ == "__main__":
    diagnose_india()
