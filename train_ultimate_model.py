import sys, os, time, math, re, unicodedata, random, joblib
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/code/business_entity_resolution"))

import polars as pl
import numpy as np
from collections import defaultdict
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier

from test_ultimate_keys import UltimateInvertedIndex, extract_ultimate_keys, clean_text_indic, get_phonetic_skeleton
from src.utils import clean_number, LEGAL_STOPWORDS, ADDR_STOPWORDS, evaluate_macro_f05, evaluate_entity_f05

def get_building_number(addr):
    if not addr: return None
    tokens = addr.replace("#", " ").replace(",", " ").split()
    for tok in tokens:
        cleaned = re.sub(r"[^0-9a-zA-Z]", "", tok)
        if re.match(r"^\d+(st|nd|rd|th)$", cleaned.lower()):
            continue
        m = re.match(r"^(\d+[\w\-\/]*)", tok)
        if m:
            num = re.sub(r"[^0-9]", "", m.group(1))
            if num: return num
    return None

def compute_all_features(n1, a1, n2, a2, raw_a1, raw_a2):
    n_fuzz = fuzz.ratio(n1, n2) / 100.0
    n_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    n_token_set = fuzz.token_set_ratio(n1, n2) / 100.0
    n_partial = fuzz.partial_ratio(n1, n2) / 100.0
    
    exact_name = 1.0 if n1 == n2 else 0.0
    
    w1 = [w for w in n1.split() if w not in LEGAL_STOPWORDS]
    w2 = [w for w in n2.split() if w not in LEGAL_STOPWORDS]
    sw1 = set(w1)
    sw2 = set(w2)
    n_jaccard = len(sw1 & sw2) / max(1, len(sw1 | sw2))
    n_overlap = float(len(sw1 & sw2))
    exact_name_no_legal = 1.0 if (w1 and w2 and w1 == w2) else 0.0
    
    first_match = 1.0 if (w1 and w2 and (w1[0] == w2[0] or (len(w1[0]) >= 4 and len(w2[0]) >= 4 and w1[0][:4] == w2[0][:4]))) else 0.0
    
    # Phonetic skeletons
    sk1 = [get_phonetic_skeleton(w) for w in w1 if get_phonetic_skeleton(w)]
    sk2 = [get_phonetic_skeleton(w) for w in w2 if get_phonetic_skeleton(w)]
    sk_sim = float(len(set(sk1) & set(sk2))) if (sk1 and sk2) else 0.0
    
    # Address
    has_a2 = bool(a2)
    addr_null = 0.0 if has_a2 else 1.0
    
    if not has_a2 or not a2:
        a_token_set = 0.0
        a_token_sort = 0.0
        a_jaccard = 0.0
        num_match = 0.5
        bldg_match = 0.5
    else:
        a_token_set = fuzz.token_set_ratio(a1, a2) / 100.0
        a_token_sort = fuzz.token_sort_ratio(a1, a2) / 100.0
        
        aw1 = set(w for w in a1.split() if w not in ADDR_STOPWORDS and len(w) >= 3)
        aw2 = set(w for w in a2.split() if w not in ADDR_STOPWORDS and len(w) >= 3)
        a_jaccard = len(aw1 & aw2) / max(1, len(aw1 | aw2))
        
        nums1 = set(clean_number(w) for w in a1.split() if any(c.isdigit() for c in w))
        nums1.discard('')
        nums2 = set(clean_number(w) for w in a2.split() if any(c.isdigit() for c in w))
        nums2.discard('')
        if not nums1 or not nums2:
            num_match = 0.5
        elif nums1 & nums2:
            num_match = 1.0
        else:
            num_match = 0.0
            
        b1 = get_building_number(raw_a1)
        b2 = get_building_number(raw_a2)
        if not b1 or not b2:
            bldg_match = 0.5
        elif b1 == b2 or b1 in b2 or b2 in b1:
            bldg_match = 1.0
        else:
            bldg_match = 0.0
            
    name_x_addr = n_token_set * (a_token_set if not addr_null else n_token_set)
    max_sim = max(n_token_set, a_token_set)
    min_sim = min(n_token_set, a_token_set)
    
    return [
        n_fuzz, n_token_sort, n_token_set, n_partial, n_jaccard, n_overlap,
        exact_name, exact_name_no_legal, first_match, sk_sim,
        addr_null, a_token_set, a_token_sort, a_jaccard, num_match, bldg_match,
        name_x_addr, max_sim, min_sim
    ]

def main():
    print("="*75)
    print("TRAINING ULTIMATE MATCHING MODEL WITH 19 CALIBRATED FEATURES")
    print("="*75)
    t0 = time.time()
    
    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
    s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
    
    n_train = 40000
    n_val = 10000
    train_gt = gt_df.slice(0, n_train)
    val_gt = gt_df.slice(n_train, n_val)
    
    train_gt_dict = {r["source1_entity_id"]: set(r["matched_entity_ids"].split(",")) if r["matched_entity_ids"] else set() for r in train_gt.iter_rows(named=True)}
    val_gt_dict = {r["source1_entity_id"]: set(r["matched_entity_ids"].split(",")) if r["matched_entity_ids"] else set() for r in val_gt.iter_rows(named=True)}
    
    needed_targets = set.union(*train_gt_dict.values()) | set.union(*val_gt_dict.values())
    
    s1_train_dict = {r["entity_id"]: r for r in s1_full.filter(pl.col("entity_id").is_in(list(train_gt_dict.keys()))).iter_rows(named=True)}
    s1_val_dict = {r["entity_id"]: r for r in s1_full.filter(pl.col("entity_id").is_in(list(val_gt_dict.keys()))).iter_rows(named=True)}
    
    s2_full = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
    s3_full = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")
    
    s2_targets = {x for x in needed_targets if x.startswith("S2-")}
    s3_targets = {x for x in needed_targets if x.startswith("S3-")}
    
    s2_pool = pl.concat([s2_full.filter(pl.col("entity_id").is_in(list(s2_targets))), s2_full.sample(n=120000, seed=42)]).unique(subset=["entity_id"])
    s3_pool = pl.concat([s3_full.filter(pl.col("entity_id").is_in(list(s3_targets))), s3_full.sample(n=120000, seed=42)]).unique(subset=["entity_id"])
    
    target_records = {}
    for r in s2_pool.iter_rows(named=True): target_records[r["entity_id"]] = r
    for r in s3_pool.iter_rows(named=True): target_records[r["entity_id"]] = r
    
    s1_clean_train = {r["entity_id"]: (clean_text_indic(r["business_name"]), clean_text_indic(r["business_address"]) if r["business_address"] else "") for r in s1_train_dict.values()}
    s1_clean_val = {r["entity_id"]: (clean_text_indic(r["business_name"]), clean_text_indic(r["business_address"]) if r["business_address"] else "") for r in s1_val_dict.values()}
    target_clean = {r["entity_id"]: (clean_text_indic(r["business_name"]), clean_text_indic(r["business_address"]) if r["business_address"] else "") for r in target_records.values()}
    
    print(f"Loaded datasets in {time.time()-t0:.2f}s")
    
    indices = {"US": UltimateInvertedIndex(max_key_len=5000), "India": UltimateInvertedIndex(max_key_len=5000)}
    targets_by_c = defaultdict(list)
    recs_by_c = defaultdict(lambda: ([], [], []))
    for tid, tr in target_records.items():
        c = tr["country"]
        targets_by_c[c].append(tid)
        if c in indices:
            recs_by_c[c][0].append(tr["entity_id"])
            recs_by_c[c][1].append(tr["business_name"])
            recs_by_c[c][2].append(tr["business_address"])
            
    for c in indices:
        eids, names, addrs = recs_by_c[c]
        print(f"Building UltimateInvertedIndex for {c} with {len(eids)} records...")
        indices[c].add_records(eids, names, addrs)
        
    print("Generating training samples (positives + hard negatives + random negatives)...")
    t1 = time.time()
    X_train = []
    y_train = []
    random.seed(42)
    
    for s1_id, targets in train_gt_dict.items():
        s1_r = s1_train_dict[s1_id]
        c = s1_r["country"]
        s1_cn, s1_ca = s1_clean_train[s1_id]
        
        # Positives
        for tid in targets:
            tr = target_records.get(tid)
            if tr:
                t_cn, t_ca = target_clean[tid]
                X_train.append(compute_all_features(s1_cn, s1_ca, t_cn, t_ca, s1_r["business_address"], tr["business_address"]))
                y_train.append(1)
                
        # Hard Negatives
        if c in indices:
            c_indices = indices[c].query(s1_r["business_name"], s1_r["business_address"], top_name=15, top_addr=10)
            cand_eids = [indices[c].records[i][0] for i in c_indices if indices[c].records[i][0] not in targets]
            for hn_id in cand_eids[:4]:
                tr = target_records[hn_id]
                t_cn, t_ca = target_clean[hn_id]
                feats = compute_all_features(s1_cn, s1_ca, t_cn, t_ca, s1_r["business_address"], tr["business_address"])
                # Skip duplicate true entity matches
                if feats[2] >= 0.90 and feats[11] >= 0.90:
                    continue
                X_train.append(feats)
                y_train.append(0)
                
        # Random Negatives
        c_pool = targets_by_c[c]
        for rn_id in random.sample(c_pool, min(2, len(c_pool))):
            if rn_id not in targets:
                tr = target_records[rn_id]
                t_cn, t_ca = target_clean[rn_id]
                X_train.append(compute_all_features(s1_cn, s1_ca, t_cn, t_ca, s1_r["business_address"], tr["business_address"]))
                y_train.append(0)
                
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"X_train ready in {time.time()-t1:.2f}s: {X_train.shape[0]} pairs (pos={sum(y_train)}, neg={len(y_train)-sum(y_train)})")
    
    print("Fitting HistGradientBoostingClassifier...")
    t2 = time.time()
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=40,
        min_samples_leaf=20,
        early_stopping=True,
        random_state=42
    )
    clf.fit(X_train, y_train)
    print(f"Fit completed in {time.time()-t2:.2f}s (iterations: {clf.n_iter_})")
    
    print("Evaluating on 10,000 Validation Entities...")
    val_cand_pairs = []
    val_cand_features = []
    
    for s1_id in val_gt_dict:
        s1_r = s1_val_dict[s1_id]
        c = s1_r["country"]
        s1_cn, s1_ca = s1_clean_val[s1_id]
        if c in indices:
            c_indices = indices[c].query(s1_r["business_name"], s1_r["business_address"], top_name=25, top_addr=15)
            cand_eids = [indices[c].records[i][0] for i in c_indices]
            for cid in cand_eids:
                if cid in target_clean:
                    tr = target_records[cid]
                    t_cn, t_ca = target_clean[cid]
                    feats = compute_all_features(s1_cn, s1_ca, t_cn, t_ca, s1_r["business_address"], tr["business_address"])
                    val_cand_features.append(feats)
                    val_cand_pairs.append((s1_id, cid))
                    
    probs = clf.predict_proba(np.array(val_cand_features, dtype=np.float32))[:, 1]
    
    best_th = 0.5
    best_f05 = 0.0
    for th in [0.70, 0.75, 0.78, 0.80, 0.82, 0.85]:
        raw_matches = defaultdict(list)
        for (s1_id, cid), p in zip(val_cand_pairs, probs):
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
                    
        scores = [evaluate_entity_f05(val_gt_dict[s1_id], final_preds.get(s1_id, set())) for s1_id in val_gt_dict]
        macro_f05 = float(np.mean(scores))
        print(f"  Th: {th:.2f} -> Validation Macro F0.5: {macro_f05:.6f}")
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_th = th
            
    print(f"\n>>> BEST SCORE: Threshold {best_th:.2f} with Macro F0.5 = {best_f05:.6f} <<<")
    
    # Save the new ultimate model artifact
    new_artifact = {
        "model": clf,
        "threshold": best_th,
        "val_f05": best_f05
    }
    save_path = "code/business_entity_resolution/src/matching_model_ultimate.joblib"
    joblib.dump(new_artifact, save_path)
    print(f"Saved new ultimate model to {save_path} ({os.path.getsize(save_path)/1024:.1f} KB)")

if __name__ == "__main__":
    main()
