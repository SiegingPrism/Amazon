import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/code/business_entity_resolution"))

import polars as pl
import numpy as np
import joblib
from collections import defaultdict
from src.features import compute_features_precleaned
from src.utils import clean_text, evaluate_entity_f05
from src.blocking import InvertedIndex, extract_blocking_keys

def diagnose():
    print("Running in-depth error diagnosis on 10,000 held-out validation entities...")
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
    
    s1_clean = {r["entity_id"]: (clean_text(r["business_name"]), clean_text(r["business_address"]) if r["business_address"] else "") for r in s1_val_df.iter_rows(named=True)}
    target_clean = {r["entity_id"]: (clean_text(r["business_name"]), clean_text(r["business_address"]) if r["business_address"] else "", bool(r["business_address"])) for r in target_records.values()}
    
    indices = {"US": InvertedIndex(max_key_len=1000), "India": InvertedIndex(max_key_len=1000)}
    for r in target_records.values():
        c = r["country"]
        if c in indices:
            indices[c].records.append((r["entity_id"], r["business_name"], r["business_address"]))
            idx_pos = len(indices[c].records) - 1
            for k in set(extract_blocking_keys(r["business_name"], r["business_address"])):
                indices[c].index[k].append(idx_pos)
                
    model_artifact = joblib.load("code/business_entity_resolution/src/matching_model.joblib")
    clf = model_artifact["model"]
    th = model_artifact["threshold"] # 0.80
    
    cand_pairs_to_score = []
    pair_tracking = []
    retrieved_cands = defaultdict(set)
    
    for s1_id in val_gt_dict:
        s1_r = s1_val_dict[s1_id]
        c = s1_r["country"]
        if c in indices:
            c_indices = indices[c].query(s1_r["business_name"], s1_r["business_address"], top_k=20)
            cand_eids = [indices[c].records[i][0] for i in c_indices]
            retrieved_cands[s1_id] = set(cand_eids)
            s1_cn, s1_ca = s1_clean[s1_id]
            for cid in cand_eids:
                if cid in target_clean:
                    t_cn, t_ca, t_ha = target_clean[cid]
                    cand_pairs_to_score.append(compute_features_precleaned(s1_cn, s1_ca, t_cn, t_ca, t_ha))
                    pair_tracking.append((s1_id, cid))
                    
    probs = clf.predict_proba(np.array(cand_pairs_to_score, dtype=np.float32))[:, 1]
    
    raw_matches = defaultdict(list)
    pair_probs = {}
    for (s1_id, cid), p in zip(pair_tracking, probs):
        pair_probs[(s1_id, cid)] = float(p)
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
                
    # Breakdown of errors:
    total_gt_matches = sum(len(v) for v in val_gt_dict.values())
    total_retrieved = sum(len(gt & retrieved_cands[s1_id]) for s1_id, gt in val_gt_dict.items())
    fn_not_retrieved = 0
    fn_below_threshold = 0
    fn_stolen = 0
    fp_count = 0
    singleton_fp = 0
    
    stolen_examples = []
    below_th_examples = []
    not_retrieved_examples = []
    fp_examples = []
    
    for s1_id, gt_ids in val_gt_dict.items():
        pred_ids = final_preds.get(s1_id, set())
        # False positives
        for cid in (pred_ids - gt_ids):
            fp_count += 1
            if len(gt_ids) == 0:
                singleton_fp += 1
            if len(fp_examples) < 8:
                s1_r = s1_val_dict[s1_id]
                tr = target_records[cid]
                fp_examples.append((s1_id, s1_r["business_name"], s1_r["business_address"],
                                    cid, tr["business_name"], tr["business_address"],
                                    pair_probs.get((s1_id, cid), 0.0)))
        # False negatives
        for cid in (gt_ids - pred_ids):
            if cid not in retrieved_cands[s1_id]:
                fn_not_retrieved += 1
                if len(not_retrieved_examples) < 8:
                    s1_r = s1_val_dict[s1_id]
                    tr = target_records.get(cid, {})
                    not_retrieved_examples.append((s1_id, s1_r["business_name"], s1_r["business_address"],
                                                   cid, tr.get("business_name",""), tr.get("business_address","")))
            else:
                prob = pair_probs.get((s1_id, cid), 0.0)
                if prob < th:
                    fn_below_threshold += 1
                    if len(below_th_examples) < 8:
                        s1_r = s1_val_dict[s1_id]
                        tr = target_records[cid]
                        below_th_examples.append((s1_id, s1_r["business_name"], s1_r["business_address"],
                                                  cid, tr["business_name"], tr["business_address"], prob))
                else:
                    # was stolen by someone else!
                    fn_stolen += 1
                    winner_s1, winner_p = target_winner.get(cid, (None, 0.0))
                    if len(stolen_examples) < 8:
                        s1_r = s1_val_dict[s1_id]
                        tr = target_records[cid]
                        win_r = s1_val_dict.get(winner_s1, {})
                        stolen_examples.append((s1_id, s1_r["business_name"], cid, tr["business_name"],
                                                prob, winner_s1, win_r.get("business_name",""), winner_p))
                        
    print("="*60)
    print("ERROR BREAKDOWN (10,000 Validation Entities):")
    print(f"Total True Matches:           {total_gt_matches}")
    print(f"Blocking Misses (Recall Loss): {fn_not_retrieved} ({fn_not_retrieved/total_gt_matches*100:.2f}%)")
    print(f"Model Misses (p < threshold):  {fn_below_threshold} ({fn_below_threshold/total_gt_matches*100:.2f}%)")
    print(f"Disambiguation Stolen:        {fn_stolen} ({fn_stolen/total_gt_matches*100:.2f}%)")
    print(f"Total False Positives:        {fp_count} (Singletons ruined: {singleton_fp})")
    print("="*60)
    
    print("\nSAMPLE BLOCKING MISSES (fn_not_retrieved):")
    for ex in not_retrieved_examples[:5]:
        print(f"  S1: {ex[1]} | {ex[2]}")
        print(f"  TR: {ex[4]} | {ex[5]}\n")
        
    print("\nSAMPLE BELOW THRESHOLD (fn_below_threshold):")
    for ex in below_th_examples[:5]:
        print(f"  S1: {ex[1]} | {ex[2]}")
        print(f"  TR: {ex[4]} | {ex[5]} (p = {ex[6]:.4f})\n")
        
    print("\nSAMPLE FALSE POSITIVES (fp_count):")
    for ex in fp_examples[:5]:
        print(f"  S1: {ex[1]} | {ex[2]}")
        print(f"  TR: {ex[4]} | {ex[5]} (p = {ex[6]:.4f})\n")
        
    print("\nSAMPLE STOLEN DISAMBIGUATION (fn_stolen):")
    for ex in stolen_examples[:5]:
        print(f"  True S1: {ex[1]} (p={ex[4]:.4f}) vs Stolen By {ex[5]} ({ex[6]}, p={ex[7]:.4f}) for Target {ex[3]}\n")

if __name__ == "__main__":
    diagnose()
