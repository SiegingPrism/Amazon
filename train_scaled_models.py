import sys, os, time, random, joblib
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/code/business_entity_resolution"))

import polars as pl
import numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from src.utils import clean_text, clean_address, evaluate_macro_f05, evaluate_entity_f05
from src.features_v2 import compute_features_v2_precleaned
from src.blocking import InvertedIndex

print("="*75)
print("TRAINING PRODUCTION-SCALE COUNTRY-SPECIALIZED MODELS (US & INDIA)")
print("="*75)

t_start = time.time()

# 1. Read datasets with Polars
print("Reading training datasets with Polars...")
t0 = time.time()
gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
s2_full = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
s3_full = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")
print(f"Loaded datasets in {time.time()-t0:.2f}s: S1={len(s1_full):,}, S2={len(s2_full):,}, S3={len(s3_full):,}, GT={len(gt_df):,}")

# Join country onto GT
gt_with_country = gt_df.join(s1_full.select(["entity_id", "country"]), left_on="source1_entity_id", right_on="entity_id")

# Train each country independently
for country, n_train, n_val in [("US", 180000, 15000), ("India", 180000, 15000)]:
    c_t0 = time.time()
    print(f"\n{'='*75}")
    print(f">>> TRAINING HIGH-PRECISION MODEL FOR: {country} <<<")
    print(f"{'='*75}")
    
    country_gt = gt_with_country.filter(pl.col("country") == country)
    print(f"Available {country} entities: {len(country_gt):,}")
    
    train_gt_slice = country_gt.slice(0, n_train)
    val_gt_slice = country_gt.slice(n_train, n_val)
    
    train_gt_dict = {}
    needed_train_targets = set()
    for r in train_gt_slice.iter_rows(named=True):
        eid = r["source1_entity_id"]
        if r["matched_entity_ids"]:
            t_set = set(r["matched_entity_ids"].split(","))
            train_gt_dict[eid] = t_set
            needed_train_targets.update(t_set)
        else:
            train_gt_dict[eid] = set()
            
    val_gt_dict = {}
    needed_val_targets = set()
    for r in val_gt_slice.iter_rows(named=True):
        eid = r["source1_entity_id"]
        if r["matched_entity_ids"]:
            t_set = set(r["matched_entity_ids"].split(","))
            val_gt_dict[eid] = t_set
            needed_val_targets.update(t_set)
        else:
            val_gt_dict[eid] = set()
            
    print(f"Train S1 entities: {len(train_gt_dict):,}, Val S1 entities: {len(val_gt_dict):,}")
    
    # Target pool for this country: All needed targets + 500,000 background records
    s2_c = s2_full.filter(pl.col("country") == country)
    s3_c = s3_full.filter(pl.col("country") == country)
    
    all_needed = needed_train_targets.union(needed_val_targets)
    s2_needed = [x for x in all_needed if x.startswith("S2-")]
    s3_needed = [x for x in all_needed if x.startswith("S3-")]
    
    s2_bg = s2_c.sample(n=min(300000, len(s2_c)), seed=42)
    s3_bg = s3_c.sample(n=min(300000, len(s3_c)), seed=42)
    
    s2_pool = pl.concat([s2_c.filter(pl.col("entity_id").is_in(s2_needed)), s2_bg]).unique(subset=["entity_id"])
    s3_pool = pl.concat([s3_c.filter(pl.col("entity_id").is_in(s3_needed)), s3_bg]).unique(subset=["entity_id"])
    print(f"Target pool assembled: S2={len(s2_pool):,}, S3={len(s3_pool):,}")
    
    # Pre-clean targets
    t_clean0 = time.time()
    target_clean = {} # eid -> (cn, ca, ha)
    for df in [s2_pool, s3_pool]:
        eids = df["entity_id"].to_list()
        names = df["business_name"].to_list()
        addrs = df["business_address"].to_list()
        c_names = [clean_text(n) for n in names]
        c_addrs = [clean_address(a, country=country) if a else "" for a in addrs]
        has_addrs = [bool(a) for a in addrs]
        for i in range(len(eids)):
            target_clean[eids[i]] = (c_names[i], c_addrs[i], has_addrs[i])
            
    # Build Inverted Index for candidate and hard negative generation
    print(f"Building Inverted Index on {len(target_clean):,} records...")
    idx = InvertedIndex(max_key_len=2000)
    e_list = list(target_clean.keys())
    idx.add_records(
        e_list,
        [target_clean[e][0] for e in e_list],
        [target_clean[e][1] for e in e_list],
        country=country
    )
    print(f"Index built in {time.time()-t_clean0:.2f}s with {len(idx.index):,} keys")
    
    # Pre-clean S1 records
    s1_c_df = s1_full.filter(pl.col("country") == country)
    s1_dict = {}
    for r in s1_c_df.filter(pl.col("entity_id").is_in(list(train_gt_dict.keys()) + list(val_gt_dict.keys()))).iter_rows(named=True):
        cn = clean_text(r["business_name"])
        ca = clean_address(r["business_address"], country=country) if r["business_address"] else ""
        ha = bool(r["business_address"])
        s1_dict[r["entity_id"]] = (cn, ca, ha)
        
    # Generate Training Feature Matrix
    print("\nMining hard negatives and assembling 21D training feature matrix...")
    t_pairs0 = time.time()
    X_train = []
    y_train = []
    random.seed(42)
    all_target_ids = list(target_clean.keys())
    
    for s1_id, targets in train_gt_dict.items():
        if s1_id not in s1_dict:
            continue
        cn1, ca1, ha1 = s1_dict[s1_id]
        
        # Positives
        for tid in targets:
            if tid in target_clean:
                cn2, ca2, ha2 = target_clean[tid]
                feats = compute_features_v2_precleaned(cn1, ca1, cn2, ca2, ha2)
                X_train.append(feats)
                y_train.append(1)
                
        # Hard Negatives from Inverted Index (name and address distractors)
        cands = idx.query_precleaned(cn1, ca1, ha1, top_name=16, top_addr=8)
        hn_count = 0
        for ci in cands:
            cid = idx.records[ci][0]
            if cid not in targets and cid in target_clean:
                cn2, ca2, ha2 = target_clean[cid]
                feats = compute_features_v2_precleaned(cn1, ca1, cn2, ca2, ha2)
                X_train.append(feats)
                y_train.append(0)
                hn_count += 1
                if hn_count >= 2:
                    break
                    
        # Random Negative (anchors base rate)
        for _ in range(1):
            rnd_id = random.choice(all_target_ids)
            if rnd_id not in targets:
                cn2, ca2, ha2 = target_clean[rnd_id]
                feats = compute_features_v2_precleaned(cn1, ca1, cn2, ca2, ha2)
                X_train.append(feats)
                y_train.append(0)
                
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    n_pos = int(np.sum(y_train))
    n_neg = len(y_train) - n_pos
    print(f"Features assembled in {time.time()-t_pairs0:.2f}s:")
    print(f"  Total Pairs: {len(y_train):,} (Pos: {n_pos:,}, Neg: {n_neg:,}, Pos Ratio: {n_pos/len(y_train):.2%})")
    print(f"  Feature Matrix Memory: {X_train.nbytes / (1024*1024):.1f} MB")
    
    # Train Deep Gradient Boosted Trees
    print(f"\nFitting deep HistGradientBoostingClassifier (max_leaf_nodes=63, l2=0.2)...")
    t_fit0 = time.time()
    clf = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.07,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=0.2,
        early_stopping=True,
        n_iter_no_change=12,
        random_state=42
    )
    clf.fit(X_train, y_train)
    print(f"Model fitted in {time.time()-t_fit0:.2f}s! (Trained trees: {clf.n_iter_})")
    
    # Fine-Grained Threshold Optimization on Validation Set
    print(f"\nEvaluating on {len(val_gt_dict):,} held-out {country} validation entities...")
    t_v0 = time.time()
    val_scored = defaultdict(list)
    val_batch_feats = []
    val_pair_tracking = []
    
    val_eids = list(val_gt_dict.keys())
    for eid in val_eids:
        if eid not in s1_dict:
            continue
        cn1, ca1, ha1 = s1_dict[eid]
        cands = idx.query_precleaned(cn1, ca1, ha1, top_name=22, top_addr=10)
        for ci in cands:
            cid = idx.records[ci][0]
            if cid in target_clean:
                cn2, ca2, ha2 = target_clean[cid]
                feats = compute_features_v2_precleaned(cn1, ca1, cn2, ca2, ha2)
                val_batch_feats.append(feats)
                val_pair_tracking.append((eid, cid, feats[8])) # feats[8] is core_token_set
                
    if val_batch_feats:
        val_probs = clf.predict_proba(np.array(val_batch_feats, dtype=np.float32))[:, 1]
        for (eid, cid, c_sim), p in zip(val_pair_tracking, val_probs):
            val_scored[eid].append((cid, float(p), c_sim))
            
    best_th = 0.90
    best_f05 = 0.0
    print("Threshold Sweep:")
    for th in [0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97]:
        raw_m = defaultdict(list)
        for eid, clist in val_scored.items():
            if not clist: continue
            best_p = max(p for _, p, _ in clist)
            for cid, p, c_sim in clist:
                if p >= th and (best_p - p) <= 0.06 and c_sim >= 0.40:
                    raw_m[eid].append((cid, p))
            if len(raw_m[eid]) > 6:
                raw_m[eid].sort(key=lambda x: -x[1])
                raw_m[eid] = raw_m[eid][:6]
        # Disambiguate
        win = {}
        for eid, m_list in raw_m.items():
            for cid, p in m_list:
                if cid not in win or p > win[cid][1]:
                    win[cid] = (eid, p)
        final_p = defaultdict(set)
        for eid, m_list in raw_m.items():
            for cid, p in m_list:
                if win.get(cid, (None,))[0] == eid:
                    final_p[eid].add(cid)
        f05_scores = [evaluate_entity_f05(val_gt_dict[eid], final_p.get(eid, set())) for eid in val_eids]
        mean_f05 = np.mean(f05_scores)
        print(f"  Threshold {th:.2f} -> Validation Macro F0.5 = {mean_f05:.5f}")
        if mean_f05 > best_f05:
            best_f05 = mean_f05
            best_th = th
            
    print(f"\n>>> {country} OPTIMAL CONFIGURATION: Threshold = {best_th:.2f} (Macro F0.5 = {best_f05:.5f}) <<<")
    
    # Save model artifact
    model_save_path = f"code/business_entity_resolution/src/model_{country}.joblib"
    joblib.dump({
        "model": clf,
        "country": country,
        "best_threshold": best_th,
        "val_f05": best_f05,
        "features": "features_v2_21D",
        "n_train": n_train,
        "n_val": n_val
    }, model_save_path)
    print(f"Saved trained {country} model artifact to: {model_save_path}")

print(f"\n{'='*75}")
print(f"ALL COUNTRY MODELS TRAINED & SAVED IN {time.time()-t_start:.2f}s!")
print(f"{'='*75}")
