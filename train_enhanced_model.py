import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/code/business_entity_resolution"))

import polars as pl
import numpy as np
import time, random, joblib
from collections import defaultdict
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier

from src.utils import clean_text, clean_number, evaluate_macro_f05, evaluate_entity_f05, LEGAL_STOPWORDS, ADDR_STOPWORDS
from test_advanced_blocking import AdvancedInvertedIndex, extract_advanced_keys, deleet_text, get_phonetic_skeleton_v2

print("="*75)
print("TRAINING ENHANCED MODEL ON 100,000 TRAINING ENTITIES")
print("="*75)

def compute_features_v2(s1_cn, s1_ca, t_cn, t_ca, has_addr2):
    # Base fuzz ratios
    n_fuzz = fuzz.ratio(s1_cn, t_cn) / 100.0
    n_token_sort = fuzz.token_sort_ratio(s1_cn, t_cn) / 100.0
    n_token_set = fuzz.token_set_ratio(s1_cn, t_cn) / 100.0
    n_partial = fuzz.partial_ratio(s1_cn, t_cn) / 100.0
    
    # Joined string ratio (recovers 412,000 domain name / unspaced records like sunriseinfotech.com)
    s1_j = s1_cn.replace(" ", "")
    t_j = t_cn.replace(" ", "")
    n_joined = fuzz.ratio(s1_j, t_j) / 100.0
    
    # Effective token set considering concatenation
    eff_n_token_set = max(n_token_set, n_joined)
    
    w1 = set(w for w in s1_cn.split() if w not in LEGAL_STOPWORDS and len(w) >= 2)
    w2 = set(w for w in t_cn.split() if w not in LEGAL_STOPWORDS and len(w) >= 2)
    n_jaccard = len(w1 & w2) / max(1, len(w1 | w2)) if (w1 or w2) else 0.0
    n_overlap = float(len(w1 & w2))
    
    addr_null = 0.0 if has_addr2 else 1.0
    if not has_addr2 or not t_ca:
        a_token_set = 0.0
        a_token_sort = 0.0
        a_jaccard = 0.0
        num_match = 0.5
    else:
        a_token_set = fuzz.token_set_ratio(s1_ca, t_ca) / 100.0
        a_token_sort = fuzz.token_sort_ratio(s1_ca, t_ca) / 100.0
        
        aw1 = set(w for w in s1_ca.split() if w not in ADDR_STOPWORDS and len(w) >= 3)
        aw2 = set(w for w in t_ca.split() if w not in ADDR_STOPWORDS and len(w) >= 3)
        a_jaccard = len(aw1 & aw2) / max(1, len(aw1 | aw2)) if (aw1 or aw2) else 0.0
        
        nums1 = set(clean_number(w) for w in s1_ca.split() if any(c.isdigit() for c in w))
        nums1.discard("")
        nums2 = set(clean_number(w) for w in t_ca.split() if any(c.isdigit() for c in w))
        nums2.discard("")
        
        if not nums1 or not nums2:
            num_match = 0.5
        elif nums1 & nums2:
            num_match = 1.0
        else:
            num_match = 0.0
            
    name_x_addr = eff_n_token_set * (a_token_set if not addr_null else eff_n_token_set)
    max_sim = max(eff_n_token_set, a_token_set)
    min_sim = min(eff_n_token_set, a_token_set)
    
    return [
        n_fuzz, n_token_sort, eff_n_token_set, n_partial, n_jaccard, n_overlap,
        n_joined, addr_null, a_token_set, a_token_sort, a_jaccard, num_match,
        name_x_addr, max_sim, min_sim
    ]

# 1. Load 100k train entities and 10k val entities
gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
s2_full = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
s3_full = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")

n_train = 100000
n_val = 10000
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

print(f"Sampling targets: {len(needed_targets)} needed targets + 200k background noise...")
s2_bg = s2_full.sample(n=100000, seed=42)
s3_bg = s3_full.sample(n=100000, seed=42)

s2_pool = pl.concat([s2_full.filter(pl.col("entity_id").is_in(list(needed_targets))), s2_bg]).unique(subset=["entity_id"])
s3_pool = pl.concat([s3_full.filter(pl.col("entity_id").is_in(list(needed_targets))), s3_bg]).unique(subset=["entity_id"])

target_records = {}
for r in s2_pool.iter_rows(named=True): target_records[r["entity_id"]] = r
for r in s3_pool.iter_rows(named=True): target_records[r["entity_id"]] = r

s1_train_dict = {r["entity_id"]: r for r in s1_full.filter(pl.col("entity_id").is_in(train_gt["source1_entity_id"].to_list())).iter_rows(named=True)}
s1_val_dict = {r["entity_id"]: r for r in s1_full.filter(pl.col("entity_id").is_in(val_gt["source1_entity_id"].to_list())).iter_rows(named=True)}

print("Pre-cleaning text...")
s1_train_clean = {r["entity_id"]: (clean_text(deleet_text(r["business_name"])), clean_text(r["business_address"]) if r["business_address"] else "") for r in s1_train_dict.values()}
s1_val_clean = {r["entity_id"]: (clean_text(deleet_text(r["business_name"])), clean_text(r["business_address"]) if r["business_address"] else "") for r in s1_val_dict.values()}
target_clean = {r["entity_id"]: (clean_text(deleet_text(r["business_name"])), clean_text(r["business_address"]) if r["business_address"] else "", bool(r["business_address"])) for r in target_records.values()}

# Build blocking index for negative mining
indices = {"US": AdvancedInvertedIndex(max_key_len=2000), "India": AdvancedInvertedIndex(max_key_len=2000)}
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

print("Building training feature vectors with hard-negative mining...")
t0 = time.time()
X_train = []
y_train = []

# Subsample 40k S1 entities for feature construction to keep train time fast (~300k pairs)
sampled_train_s1 = list(train_gt_dict.keys())[:40000]

for s1_id in sampled_train_s1:
    targets = train_gt_dict[s1_id]
    s1_r = s1_train_dict[s1_id]
    c = s1_r["country"]
    s1_cn, s1_ca = s1_train_clean[s1_id]
    
    # 1. Positives
    for tid in targets:
        if tid in target_clean:
            t_cn, t_ca, t_ha = target_clean[tid]
            X_train.append(compute_features_v2(s1_cn, s1_ca, t_cn, t_ca, t_ha))
            y_train.append(1)
            
    # 2. Hard negatives from index
    if c in indices:
        c_indices = indices[c].query(s1_r["business_name"], s1_r["business_address"], top_name=10, top_addr=5)
        for idx in c_indices:
            cid = indices[c].records[idx][0]
            if cid not in targets and cid in target_clean:
                t_cn, t_ca, t_ha = target_clean[cid]
                X_train.append(compute_features_v2(s1_cn, s1_ca, t_cn, t_ca, t_ha))
                y_train.append(0)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
print(f"X_train generated in {time.time()-t0:.2f}s: shape {X_train.shape}, {sum(y_train)} pos, {len(y_train)-sum(y_train)} neg")

# Train classifier
print("Training HistGradientBoostingClassifier...")
clf = HistGradientBoostingClassifier(
    max_iter=250,
    learning_rate=0.07,
    max_leaf_nodes=41,
    min_samples_leaf=25,
    l2_regularization=0.1,
    early_stopping=True,
    random_state=42
)
t_fit = time.time()
clf.fit(X_train, y_train)
print(f"Fit completed in {time.time()-t_fit:.2f}s (n_iter={clf.n_iter_})")

# Evaluate on 10,000 Validation Entities
print("Evaluating on 10,000 validation entities with advanced blocking & bipartite disambiguation...")
val_pairs = []
val_feats = []
val_countries = []

for s1_id in val_gt_dict:
    s1_r = s1_val_dict[s1_id]
    c = s1_r["country"]
    s1_cn, s1_ca = s1_val_clean[s1_id]
    if c in indices:
        c_indices = indices[c].query(s1_r["business_name"], s1_r["business_address"], top_name=25, top_addr=15)
        cand_eids = [indices[c].records[i][0] for i in c_indices]
        for cid in cand_eids:
            if cid in target_clean:
                t_cn, t_ca, t_ha = target_clean[cid]
                val_feats.append(compute_features_v2(s1_cn, s1_ca, t_cn, t_ca, t_ha))
                val_pairs.append((s1_id, cid))
                val_countries.append(c)

probs = clf.predict_proba(np.array(val_feats, dtype=np.float32))[:, 1]

# Calibrate thresholds
for th_a, th_na in [(0.80, 0.82), (0.82, 0.84), (0.84, 0.84), (0.85, 0.82)]:
    raw_matches = defaultdict(list)
    for (s1_id, cid), p, c, feat in zip(val_pairs, probs, val_countries, val_feats):
        has_addr = feat[7] == 0.0 # feat[7] is addr_null
        th = th_a if has_addr else th_na
        
        # Hard multi-tenant conflict guard
        num_m = feat[11]
        n_sim = feat[2] # eff_n_token_set
        if num_m == 0.0 and n_sim < 0.45: continue
        if n_sim < 0.20 and (num_m != 1.0 or feat[8] < 0.95): continue
        
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

    scs = [evaluate_entity_f05(val_gt_dict[s1_id], final_preds.get(s1_id, set())) for s1_id in val_gt_dict]
    print(f"th_addr={th_a:.2f}, th_no_addr={th_na:.2f} -> Validation Macro F0.5: {np.mean(scs):.6f}")

# Save enhanced model artifact
enhanced_artifact = {
    "model": clf,
    "features_version": "v2",
    "threshold_addr": 0.82,
    "threshold_no_addr": 0.82
}
joblib.dump(enhanced_artifact, "code/business_entity_resolution/src/matching_model_v2.joblib")
print("Saved matching_model_v2.joblib successfully!")
