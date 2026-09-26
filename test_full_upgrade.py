import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/code/business_entity_resolution"))

import polars as pl
import numpy as np
import time, math, joblib, re
from collections import defaultdict
from rapidfuzz import fuzz

from src.utils import clean_text, clean_number, evaluate_macro_f05, evaluate_entity_f05, LEGAL_STOPWORDS, ADDR_STOPWORDS, STATE_MAP

LEET_MAP = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b"}
def deleet_token(tok):
    if re.search(r"[a-zA-Z]", tok) and re.search(r"[0-9]", tok):
        return "".join(LEET_MAP.get(c, c) for c in tok)
    return tok
def deleet_text(text):
    if not text: return ""
    return " ".join(deleet_token(t) for t in text.split())

def get_phonetic_skeleton_v2(word):
    if len(word) < 2: return ""
    w = word.lower().replace("ph", "f").replace("ee", "i").replace("oo", "u")
    tr = str.maketrans({"g": "k", "d": "t", "b": "p", "v": "w", "z": "s", "j": "s", "c": "k", "q": "k", "x": "ks"})
    w = w.translate(tr)
    w = re.sub(r"(?<=[bcdfghjklmnpqrstvwxyz])h", "", w)
    sk = re.sub(r"[aeiouy\s]", "", w)
    sk = re.sub(r"(.)\1+", r"\1", sk)
    return sk if len(sk) >= 2 else ""

def clean_text_v2(text):
    return clean_text(deleet_text(text))

def compute_features_v2(n1_clean, a1_clean, n2_clean, a2_clean, has_addr2):
    n1_j = n1_clean.replace(" ", "")
    n2_j = n2_clean.replace(" ", "")
    n_joined_ratio = fuzz.ratio(n1_j, n2_j) / 100.0
    
    n_fuzz_ratio = max(fuzz.ratio(n1_clean, n2_clean) / 100.0, n_joined_ratio)
    n_token_sort = max(fuzz.token_sort_ratio(n1_clean, n2_clean) / 100.0, n_joined_ratio)
    n_token_set = max(fuzz.token_set_ratio(n1_clean, n2_clean) / 100.0, n_joined_ratio)
    n_partial = fuzz.partial_ratio(n1_clean, n2_clean) / 100.0
    
    w1 = set(w for w in n1_clean.split() if w not in LEGAL_STOPWORDS and len(w) >= 2)
    w2 = set(w for w in n2_clean.split() if w not in LEGAL_STOPWORDS and len(w) >= 2)
    
    if n1_j == n2_j or n_joined_ratio >= 0.95:
        n_jaccard = 1.0
        n_overlap = float(max(len(w1), len(w2), 1))
    else:
        n_jaccard = len(w1 & w2) / max(1, len(w1 | w2))
        n_overlap = float(len(w1 & w2))
        
    addr_null = 0.0 if has_addr2 else 1.0
    if not has_addr2 or not a2_clean:
        a_token_set = 0.0
        a_token_sort = 0.0
        a_jaccard = 0.0
        num_match = 0.5
    else:
        a_token_set = fuzz.token_set_ratio(a1_clean, a2_clean) / 100.0
        a_token_sort = fuzz.token_sort_ratio(a1_clean, a2_clean) / 100.0
        
        aw1 = set(w for w in a1_clean.split() if w not in ADDR_STOPWORDS and len(w) >= 3)
        aw2 = set(w for w in a2_clean.split() if w not in ADDR_STOPWORDS and len(w) >= 3)
        a_jaccard = len(aw1 & aw2) / max(1, len(aw1 | aw2)) if (aw1 or aw2) else 0.0
        
        nums1 = set(clean_number(w) for w in a1_clean.split() if any(c.isdigit() for c in w))
        nums1.discard("")
        nums2 = set(clean_number(w) for w in a2_clean.split() if any(c.isdigit() for c in w))
        nums2.discard("")
        
        if not nums1 or not nums2:
            num_match = 0.5
        elif nums1 & nums2:
            num_match = 1.0
        else:
            num_match = 0.0
            
    name_x_addr = n_token_set * (a_token_set if not addr_null else n_token_set)
    max_sim = max(n_token_set, a_token_set)
    min_sim = min(n_token_set, a_token_set)
    
    return [
        n_fuzz_ratio, n_token_sort, n_token_set, n_partial, n_jaccard, n_overlap,
        addr_null, a_token_set, a_token_sort, a_jaccard, num_match,
        name_x_addr, max_sim, min_sim
    ]

# Load model
model_artifact = joblib.load("code/business_entity_resolution/src/matching_model.joblib")
clf = model_artifact["model"]

print("="*75)
print("TESTING UPGRADED PIPELINE ON 10,000 VALIDATION ENTITIES")
print("="*75)

from test_advanced_blocking import AdvancedInvertedIndex

gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t").slice(10000, 10000)
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

s1_clean = {r["entity_id"]: (clean_text_v2(r["business_name"]), clean_text(r["business_address"]) if r["business_address"] else "") for r in s1_dict.values()}
target_clean = {r["entity_id"]: (clean_text_v2(r["business_name"]), clean_text(r["business_address"]) if r["business_address"] else "", bool(r["business_address"])) for r in target_records.values()}

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

val_pairs = []
val_feats = []
val_countries = []

for s1_id in gt_dict:
    s1_r = s1_dict[s1_id]
    c = s1_r["country"]
    s1_cn, s1_ca = s1_clean[s1_id]
    if c in indices:
        c_indices = indices[c].query(s1_r["business_name"], s1_r["business_address"], top_name=22, top_addr=10)
        cand_eids = [indices[c].records[i][0] for i in c_indices]
        for cid in cand_eids:
            if cid in target_clean:
                t_cn, t_ca, t_ha = target_clean[cid]
                val_feats.append(compute_features_v2(s1_cn, s1_ca, t_cn, t_ca, t_ha))
                val_pairs.append((s1_id, cid))
                val_countries.append(c)

probs = clf.predict_proba(np.array(val_feats, dtype=np.float32))[:, 1]
print(f"Total candidate pairs: {len(val_pairs)}")

# Production thresholds
raw_matches = defaultdict(list)
for (s1_id, cid), p, c, feat in zip(val_pairs, probs, val_countries, val_feats):
    has_addr = feat[6] == 0.0
    if c == "India":
        th = 0.86 if has_addr else 0.78
    else:
        th = 0.80 if has_addr else 0.82
        
    # Multi-tenant conflict guard
    num_m = feat[10]
    n_sim = feat[2]
    if num_m == 0.0 and n_sim < 0.45: continue
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

c_scores = defaultdict(list)
for s1_id in gt_dict:
    gt = gt_dict[s1_id]
    pred = final_preds.get(s1_id, set())
    sc = evaluate_entity_f05(gt, pred)
    c = s1_dict[s1_id]["country"]
    c_scores[c].append(sc)

print("\n--- RESULTS OF UPGRADED PIPELINE ---")
all_sc = []
for c, scs in c_scores.items():
    print(f"[{c}] Macro F0.5 ({len(scs)} entities): {np.mean(scs):.6f}")
    all_sc.extend(scs)
print(f"\n>>> OVERALL MACRO F0.5: {np.mean(all_sc):.6f} <<<")
