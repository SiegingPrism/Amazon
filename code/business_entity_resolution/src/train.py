import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import polars as pl
import numpy as np
import random, time, joblib
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier

from src.utils import evaluate_macro_f05, clean_text
from src.features import compute_features_precleaned
from src.blocking import InvertedIndex

def main():
    print("="*70)
    print("Training Scaled Business Entity Resolution Model (200K S1 Entities)")
    print("="*70)
    t0 = time.time()
    
    # 1. Load training ground truth
    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
    s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
    
    # Scaled training: 200,000 train S1 and 25,000 val S1
    n_train = 200000
    n_val = 25000
    print(f"Sampling {n_train:,} training S1 entities and {n_val:,} validation S1 entities...")
    train_gt = gt_df.slice(0, n_train)
    val_gt = gt_df.slice(n_train, n_val)
    
    train_gt_dict = {}
    needed_targets = set()
    for r in train_gt.iter_rows(named=True):
        s1_id = r["source1_entity_id"]
        if r["matched_entity_ids"]:
            t = set(r["matched_entity_ids"].split(","))
            train_gt_dict[s1_id] = t
            needed_targets.update(t)
        else:
            train_gt_dict[s1_id] = set()
            
    val_gt_dict = {}
    for r in val_gt.iter_rows(named=True):
        s1_id = r["source1_entity_id"]
        if r["matched_entity_ids"]:
            t = set(r["matched_entity_ids"].split(","))
            val_gt_dict[s1_id] = t
            needed_targets.update(t)
        else:
            val_gt_dict[s1_id] = set()
            
    all_needed_s1 = set(train_gt_dict.keys()).union(set(val_gt_dict.keys()))
    s1_sub = s1_full.filter(pl.col("entity_id").is_in(list(all_needed_s1)))
    
    s1_train_dict = {}
    s1_val_dict = {}
    for r in s1_sub.iter_rows(named=True):
        eid = r["entity_id"]
        if eid in train_gt_dict:
            s1_train_dict[eid] = r
        elif eid in val_gt_dict:
            s1_val_dict[eid] = r
            
    print(f"Loaded {len(s1_train_dict):,} train S1 and {len(s1_val_dict):,} val S1 in {time.time()-t0:.2f}s")
    
    # 2. Build 1.7M Target Pool (All needed targets + 500K S2 bg + 500K S3 bg)
    print("Loading S2 and S3 target records with 500K background sampling...")
    t_pool0 = time.time()
    s2_full = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
    s3_full = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")
    
    s2_targets = [x for x in needed_targets if x.startswith("S2-")]
    s3_targets = [x for x in needed_targets if x.startswith("S3-")]
    
    s2_bg = s2_full.sample(n=500000, seed=42)
    s3_bg = s3_full.sample(n=500000, seed=42)
    
    s2_pool = pl.concat([s2_full.filter(pl.col("entity_id").is_in(s2_targets)), s2_bg]).unique(subset=["entity_id"])
    s3_pool = pl.concat([s3_full.filter(pl.col("entity_id").is_in(s3_targets)), s3_bg]).unique(subset=["entity_id"])
    
    # Pre-clean all target strings for high-throughput feature extraction
    clean_targets = {} # eid -> (cn, ca, ha, country)
    target_by_country = {"US": {"ids": [], "names": [], "addrs": []},
                         "India": {"ids": [], "names": [], "addrs": []}}
    
    for df in [s2_pool, s3_pool]:
        for r in df.iter_rows(named=True):
            eid = r["entity_id"]
            c = r["country"]
            raw_name = r["business_name"]
            raw_addr = r["business_address"]
            cn = clean_text(raw_name)
            ca = clean_text(raw_addr) if raw_addr else ""
            ha = bool(raw_addr)
            clean_targets[eid] = (cn, ca, ha, c)
            if c in target_by_country:
                target_by_country[c]["ids"].append(eid)
                target_by_country[c]["names"].append(raw_name)
                target_by_country[c]["addrs"].append(raw_addr)
                
    print(f"Target pool assembled: {len(clean_targets):,} records in {time.time()-t_pool0:.2f}s")
    
    # Pre-clean S1 records
    clean_s1_train = {}
    for eid, r in s1_train_dict.items():
        cn = clean_text(r["business_name"])
        ca = clean_text(r["business_address"]) if r["business_address"] else ""
        ha = bool(r["business_address"])
        clean_s1_train[eid] = (cn, ca, ha, r["country"])
        
    clean_s1_val = {}
    for eid, r in s1_val_dict.items():
        cn = clean_text(r["business_name"])
        ca = clean_text(r["business_address"]) if r["business_address"] else ""
        ha = bool(r["business_address"])
        clean_s1_val[eid] = (cn, ca, ha, r["country"])
        
    # 3. Build Inverted Indices for Hard Negative Mining (max_key_len=2000)
    print("Building Inverted Indices per country for realistic hard negative mining...")
    t_idx0 = time.time()
    indices = {"US": InvertedIndex(max_key_len=2000), "India": InvertedIndex(max_key_len=2000)}
    for c in ["US", "India"]:
        indices[c].add_records(target_by_country[c]["ids"], target_by_country[c]["names"], target_by_country[c]["addrs"])
        print(f"  {c} Index: {len(indices[c].records):,} records, {len(indices[c].index):,} keys")
    print(f"Indices built in {time.time()-t_idx0:.2f}s")
    
    # 4. Generate Balanced Training Pairs
    print("Mining hard negatives and constructing feature matrix X_train...")
    t_feat0 = time.time()
    X_train = []
    y_train = []
    
    random.seed(42)
    c_ids_pool = {c: target_by_country[c]["ids"] for c in ["US", "India"]}
    
    for s1_id, targets in train_gt_dict.items():
        cn1, ca1, ha1, c = clean_s1_train[s1_id]
        
        # Positives
        for tid in targets:
            if tid in clean_targets:
                cn2, ca2, ha2, _ = clean_targets[tid]
                feats = compute_features_precleaned(cn1, ca1, cn2, ca2, ha2)
                X_train.append(feats)
                y_train.append(1)
                
        # Hard Negatives from Inverted Index
        if c in indices:
            cand_indices = indices[c].query_precleaned(cn1, ca1, ha1, top_name=15, top_addr=8)
            hn_count = 0
            for ci in cand_indices:
                cand_id = indices[c].records[ci][0]
                if cand_id not in targets:
                    cn2, ca2, ha2, _ = clean_targets[cand_id]
                    feats = compute_features_precleaned(cn1, ca1, cn2, ca2, ha2)
                    X_train.append(feats)
                    y_train.append(0)
                    hn_count += 1
                    if hn_count >= 3:
                        break
                        
        # Random Negative
        if c in c_ids_pool and c_ids_pool[c]:
            rn_id = random.choice(c_ids_pool[c])
            if rn_id not in targets:
                cn2, ca2, ha2, _ = clean_targets[rn_id]
                feats = compute_features_precleaned(cn1, ca1, cn2, ca2, ha2)
                X_train.append(feats)
                y_train.append(0)
                
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    n_pos = int(np.sum(y_train))
    n_neg = len(y_train) - n_pos
    print(f"Feature matrix built in {time.time()-t_feat0:.2f}s:")
    print(f"  Total Pairs: {len(y_train):,} (Positives: {n_pos:,}, Negatives: {n_neg:,}, Ratio: {n_pos/max(1,n_neg):.2f})")
    
    # 5. Fit HistGradientBoostingClassifier with L2 Regularization
    print("\nFitting HistGradientBoostingClassifier (l2_regularization=0.1, max_leaf_nodes=31)...")
    t_clf0 = time.time()
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1, # Issue 6 fix
        early_stopping=True,
        random_state=42
    )
    clf.fit(X_train, y_train)
    print(f"Model fitted in {time.time()-t_clf0:.2f}s (n_iter={clf.n_iter_})")
    
    # 6. Evaluate on 25,000 Validation Entities with Fine-Grained Threshold Sweep
    print(f"\nEvaluating on {len(val_gt_dict):,} Validation Entities with Fine-Grained Grid...")
    t_val0 = time.time()
    
    val_pred_scores = defaultdict(list)
    val_has_addr = {}
    
    for s1_id, targets in val_gt_dict.items():
        cn1, ca1, ha1, c = clean_s1_val[s1_id]
        val_has_addr[s1_id] = ha1
        
        cand_indices = []
        if c in indices:
            cand_indices = indices[c].query_precleaned(cn1, ca1, ha1, top_name=20, top_addr=10)
            
        cand_feats = []
        cand_meta = []
        for ci in cand_indices:
            cand_id = indices[c].records[ci][0]
            if cand_id in clean_targets:
                cn2, ca2, ha2, _ = clean_targets[cand_id]
                feats = compute_features_precleaned(cn1, ca1, cn2, ca2, ha2)
                cand_feats.append(feats)
                cand_meta.append((cand_id, ha2, feats[10], feats[2], feats[7])) # (cid, ha2, num_m, n_sim, a_token_set)
                
        if cand_feats:
            probs = clf.predict_proba(np.array(cand_feats, dtype=np.float32))[:, 1]
            for (cid, ha2, num_m, n_sim, a_set), p in zip(cand_meta, probs):
                # Multi-tenant conflict guard
                if num_m == 0.0 and n_sim < 0.45:
                    continue
                if n_sim < 0.20 and (num_m != 1.0 or a_set < 0.95):
                    continue
                val_pred_scores[s1_id].append((cid, float(p), ha2))
                
    print(f"Validation predictions generated in {time.time()-t_val0:.2f}s")
    
    # Fine-grained threshold sweep (0.50 to 0.95 with step 0.01)
    best_th = 0.80
    best_f05 = 0.0
    print("\n--- Fine-Grained Global Threshold Sweep ---")
    for th in np.arange(0.65, 0.92, 0.01):
        preds = {}
        for s1_id in val_gt_dict:
            matched = set(cid for cid, p, _ in val_pred_scores.get(s1_id, []) if p >= th)
            preds[s1_id] = matched
        score = evaluate_macro_f05(val_gt_dict, preds)
        if score > best_f05:
            best_f05 = score
            best_th = float(th)
        if round(th, 3) in [0.70, 0.75, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]:
            print(f"  Threshold {th:.2f} -> Validation Macro F_0.5: {score:.5f}")
            
    print(f"Optimal Global Threshold: {best_th:.2f} with Macro F_0.5 = {best_f05:.5f}")
    
    # Per-Country Validation Sweep
    print("\n--- Per-Country Validation Analysis ---")
    for country in ["US", "India"]:
        c_val_gt = {k: v for k, v in val_gt_dict.items() if clean_s1_val[k][3] == country}
        c_best_th = 0.80
        c_best_f05 = 0.0
        for th in np.arange(0.70, 0.92, 0.01):
            preds = {}
            for s1_id in c_val_gt:
                matched = set(cid for cid, p, _ in val_pred_scores.get(s1_id, []) if p >= th)
                preds[s1_id] = matched
            score = evaluate_macro_f05(c_val_gt, preds)
            if score > c_best_f05:
                c_best_f05 = score
                c_best_th = float(th)
        print(f"  Country {country:5s} ({len(c_val_gt):,} entities) -> Best Threshold: {c_best_th:.2f}, Macro F_0.5: {c_best_f05:.5f}")
        
    # Dual Threshold Sweep (th_addr vs th_no_addr)
    print("\n--- Dual Threshold Calibration (With-Address vs Missing-Address) ---")
    best_dual_f05 = 0.0
    best_pair = (0.80, 0.80)
    for tha in [0.78, 0.80, 0.82, 0.84, 0.86]:
        for thna in [0.76, 0.78, 0.80, 0.82, 0.84]:
            preds = {}
            for s1_id in val_gt_dict:
                matched = set(cid for cid, p, ha2 in val_pred_scores.get(s1_id, []) if p >= (tha if ha2 else thna))
                preds[s1_id] = matched
            score = evaluate_macro_f05(val_gt_dict, preds)
            if score > best_dual_f05:
                best_dual_f05 = score
                best_pair = (tha, thna)
    print(f"Optimal Dual Thresholds: th_addr={best_pair[0]:.2f}, th_no_addr={best_pair[1]:.2f} -> Macro F_0.5: {best_dual_f05:.5f}")
    
    # 7. Save Retrained Model Artifact
    model_artifact = {
        "model": clf,
        "threshold": best_th,
        "val_f05": best_f05,
        "best_dual": best_pair,
        "n_train": n_train,
        "n_val": n_val,
        "train_time": time.time() - t0
    }
    model_path = os.path.join(os.path.dirname(__file__), "matching_model.joblib")
    joblib.dump(model_artifact, model_path)
    print(f"\nRetrained model artifact saved to {model_path} ({os.path.getsize(model_path) / 1024:.1f} KB)")
    print(f"Total Workflow Completed in {time.time()-t0:.2f}s ({(time.time()-t0)/60:.2f} minutes)!")

if __name__ == "__main__":
    main()
