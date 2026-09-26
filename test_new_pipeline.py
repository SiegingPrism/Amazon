import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/code/business_entity_resolution"))

import polars as pl
import numpy as np
import random, time, math, joblib, re
from collections import defaultdict
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier

from src.utils import clean_text, clean_number, LEGAL_STOPWORDS, ADDR_STOPWORDS, evaluate_macro_f05, evaluate_entity_f05

STATE_MAP = {
    # US
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas', 'ca': 'california',
    'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware', 'fl': 'florida', 'ga': 'georgia',
    'hi': 'hawaii', 'id': 'idaho', 'il': 'illinois', 'in': 'indiana', 'ia': 'iowa',
    'ks': 'kansas', 'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi', 'mo': 'missouri',
    'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada', 'nh': 'new hampshire', 'nj': 'new jersey',
    'nm': 'new mexico', 'ny': 'new york', 'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio',
    'ok': 'oklahoma', 'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
    'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah', 'vt': 'vermont',
    'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia', 'wi': 'wisconsin', 'wy': 'wyoming',
    # India
    'ap': 'andhra pradesh', 'ar': 'arunachal pradesh', 'as': 'assam', 'br': 'bihar',
    'cg': 'chhattisgarh', 'ct': 'chhattisgarh', 'ga': 'goa', 'gj': 'gujarat',
    'hr': 'haryana', 'hp': 'himachal pradesh', 'jh': 'jharkhand', 'ka': 'karnataka',
    'kl': 'kerala', 'mp': 'madhya pradesh', 'mh': 'maharashtra', 'mn': 'manipur',
    'ml': 'meghalaya', 'mz': 'mizoram', 'nl': 'nagaland', 'od': 'odisha',
    'or': 'odisha', 'pb': 'punjab', 'rj': 'rajasthan', 'sk': 'sikkim',
    'tn': 'tamil nadu', 'tg': 'telangana', 'ts': 'telangana', 'tr': 'tripura',
    'up': 'uttar pradesh', 'uk': 'uttarakhand', 'ut': 'uttarakhand', 'wb': 'west bengal',
    'dl': 'delhi', 'jk': 'jammu kashmir', 'la': 'ladakh', 'py': 'puducherry', 'ch': 'chandigarh'
}

def extract_improved_keys(name, address):
    clean_n = clean_text(name)
    clean_a = clean_text(address) if address else ''
    has_addr = bool(address)
    keys = []
    raw_words = [w for w in clean_n.split() if w not in LEGAL_STOPWORDS]
    words = [w for w in raw_words if len(w) >= 3]
    if not words and raw_words: words = raw_words
        
    if words:
        keys.append('NF:' + '_'.join(words[:3]))
        for w in words[:4]:
            if len(w) >= 3:
                keys.append('NW:' + w)
                if len(w) >= 5:
                    keys.append('NP:' + w[:4])
            elif len(w) == 2:
                keys.append('NW2:' + w)
                
    if has_addr and clean_a:
        a_tokens = clean_a.split()
        expanded_tokens = []
        for tok in a_tokens:
            if tok in STATE_MAP:
                expanded_tokens.extend(STATE_MAP[tok].split())
            else:
                expanded_tokens.append(tok)
                
        nums = [clean_number(w) for w in expanded_tokens if any(c.isdigit() for c in w)]
        nums = [n for n in nums if n]
        awords = [w for w in expanded_tokens if w not in ADDR_STOPWORDS and not any(c.isdigit() for c in w) and len(w) >= 3]
        
        if nums and awords:
            sig_words = list(dict.fromkeys(awords[:3] + awords[-3:]))
            for num in nums[:2]:
                for aw in sig_words:
                    keys.append(f'ANW:{num}_{aw}')
                    
        if len(awords) >= 2:
            keys.append(f'AAW:{awords[0]}_{awords[1]}')
            keys.append(f'AAW:{awords[-2]}_{awords[-1]}')
            if len(awords) >= 3:
                keys.append(f'AAW:{awords[0]}_{awords[-1]}')
                
        if words and awords:
            keys.append(f'NAW:{words[0]}_{awords[0]}')
            if len(awords) >= 2:
                keys.append(f'NAW:{words[0]}_{awords[-1]}')
                
    return keys

def compute_enhanced_features(s1_n, s1_a, s2_n, s2_a):
    n1 = clean_text(s1_n)
    n2 = clean_text(s2_n)
    a1 = clean_text(s1_a) if s1_a else ''
    a2 = clean_text(s2_a) if s2_a else ''
    
    # 1. Base name metrics
    n_fuzz = fuzz.ratio(n1, n2) / 100.0
    n_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    n_token_set = fuzz.token_set_ratio(n1, n2) / 100.0
    n_partial = fuzz.partial_ratio(n1, n2) / 100.0
    
    w1 = [w for w in n1.split() if w not in LEGAL_STOPWORDS]
    w2 = [w for w in n2.split() if w not in LEGAL_STOPWORDS]
    sw1 = set(w1)
    sw2 = set(w2)
    n_jaccard = len(sw1 & sw2) / max(1, len(sw1 | sw2))
    n_overlap = float(len(sw1 & sw2))
    
    # First word match
    first_match = 1.0 if (w1 and w2 and (w1[0] == w2[0] or (len(w1[0]) >= 4 and len(w2[0]) >= 4 and w1[0][:4] == w2[0][:4]))) else 0.0
    
    # Phonetic skeleton similarity
    def p_skel(s):
        s = s.replace('ph', 'f').replace('ee', 'i').replace('oo', 'u')
        tr = str.maketrans({'g': 'k', 'd': 't', 'b': 'p', 'v': 'w', 'z': 's', 'j': 's', 'c': 's', 'x': 's', 'q': 'k'})
        s = s.translate(tr)
        s = re.sub(r'([bcdfghjklmnpqrstvwxyz])h', r'\1', s)
        s = re.sub(r'(.)\1+', r'\1', s)
        return re.sub(r'[aeiou\s]', '', s)
    sk1 = p_skel(n1)
    sk2 = p_skel(n2)
    p_sim = fuzz.ratio(sk1, sk2) / 100.0 if (sk1 and sk2) else n_fuzz
    
    # 2. Address metrics
    has_a = bool(s2_a and a2)
    addr_null = 0.0 if has_a else 1.0
    if not has_a:
        a_token_set = 0.0
        a_token_sort = 0.0
        a_jaccard = 0.0
        num_match = 0.5
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
            
    name_x_addr = n_token_set * (a_token_set if not addr_null else n_token_set)
    max_sim = max(n_token_set, a_token_set)
    min_sim = min(n_token_set, a_token_set)
    
    return [
        n_fuzz, n_token_sort, n_token_set, n_partial, n_jaccard, n_overlap,
        first_match, p_sim,
        addr_null, a_token_set, a_token_sort, a_jaccard, num_match,
        name_x_addr, max_sim, min_sim
    ]

class FastInvertedIndex:
    def __init__(self, max_key_len=3000):
        self.index = defaultdict(list)
        self.max_key_len = max_key_len
        self.records = []
        self.idfs = {}

    def add_records(self, eids, names, addrs):
        start_idx = len(self.records)
        for i in range(len(eids)):
            idx = start_idx + i
            self.records.append((eids[i], names[i], addrs[i]))
            keys = set(extract_improved_keys(names[i], addrs[i]))
            for k in keys:
                self.index[k].append(idx)
        # Compute IDF
        N = len(self.records)
        for k, posting in self.index.items():
            self.idfs[k] = math.log(1.0 + (N / len(posting)))

    def query(self, name, address, top_name=18, top_addr=12):
        keys = set(extract_improved_keys(name, address))
        name_scores = {}
        addr_scores = {}
        for k in keys:
            posting = self.index.get(k)
            if posting and len(posting) <= self.max_key_len:
                idf = self.idfs.get(k, 1.0)
                if k.startswith(('NF:', 'NW:', 'NW2:', 'NP:', 'NAW:')):
                    mult = 2.0 if k.startswith('NF:') else 1.0
                    for i in posting:
                        name_scores[i] = name_scores.get(i, 0.0) + (idf * mult)
                if k.startswith(('ANW:', 'AAW:', 'NAW:')):
                    mult = 1.5 if k.startswith('ANW:') else 1.0
                    for i in posting:
                        addr_scores[i] = addr_scores.get(i, 0.0) + (idf * mult)
                        
        top_n = sorted(name_scores, key=name_scores.get, reverse=True)[:top_name]
        top_a = sorted(addr_scores, key=addr_scores.get, reverse=True)[:top_addr]
        return list(dict.fromkeys(top_n + top_a))

def main():
    print("="*60)
    print("Training Next-Gen High Accuracy Model (Target: > 0.985 F0.5)")
    print("="*60)
    t0 = time.time()
    
    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
    s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
    
    n_train = 50000
    n_val = 5000
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
            
    s1_train_dict = {r["entity_id"]: r for r in s1_full.filter(pl.col("entity_id").is_in(train_gt["source1_entity_id"].to_list())).iter_rows(named=True)}
    s1_val_dict = {r["entity_id"]: r for r in s1_full.filter(pl.col("entity_id").is_in(val_gt["source1_entity_id"].to_list())).iter_rows(named=True)}
    
    s2_full = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
    s3_full = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")
    
    s2_targets = {x for x in needed_targets if x.startswith("S2-")}
    s3_targets = {x for x in needed_targets if x.startswith("S3-")}
    
    # Pool with 120k background samples
    s2_pool = pl.concat([s2_full.filter(pl.col("entity_id").is_in(list(s2_targets))), s2_full.sample(n=120000, seed=42)]).unique(subset=["entity_id"])
    s3_pool = pl.concat([s3_full.filter(pl.col("entity_id").is_in(list(s3_targets))), s3_full.sample(n=120000, seed=42)]).unique(subset=["entity_id"])
    
    target_records = {}
    for r in s2_pool.iter_rows(named=True): target_records[r["entity_id"]] = r
    for r in s3_pool.iter_rows(named=True): target_records[r["entity_id"]] = r
    
    print(f"Loaded datasets in {time.time()-t0:.2f}s")
    
    # Batch Inverted Indexes per country
    indices = {"US": FastInvertedIndex(max_key_len=3000), "India": FastInvertedIndex(max_key_len=3000)}
    targets_by_country = defaultdict(list)
    recs_by_c = defaultdict(lambda: ([], [], []))
    for tid, tr in target_records.items():
        c = tr["country"]
        targets_by_country[c].append(tid)
        if c in indices:
            recs_by_c[c][0].append(tr["entity_id"])
            recs_by_c[c][1].append(tr["business_name"])
            recs_by_c[c][2].append(tr["business_address"])
            
    for c in indices:
        eids, names, addrs = recs_by_c[c]
        print(f"Building index for {c} with {len(eids)} records...")
        indices[c].add_records(eids, names, addrs)
            
    print("Generating enhanced training pairs (positives, address-colocated hard negatives, and random negatives)...")
    t1 = time.time()
    X_train = []
    y_train = []
    
    random.seed(42)
    for s1_id, targets in train_gt_dict.items():
        s1_r = s1_train_dict[s1_id]
        c = s1_r["country"]
        
        # Positives
        for tid in targets:
            tr = target_records.get(tid)
            if tr:
                X_train.append(compute_enhanced_features(s1_r["business_name"], s1_r["business_address"], tr["business_name"], tr["business_address"]))
                y_train.append(1)
                
        # Clean Hard Negatives
        if c in indices:
            c_cand_idx = indices[c].query(s1_r["business_name"], s1_r["business_address"], top_name=10, top_addr=8)
            cand_eids = [indices[c].records[i][0] for i in c_cand_idx if indices[c].records[i][0] not in targets]
            for hn_id in cand_eids[:3]:
                tr = target_records[hn_id]
                feats = compute_enhanced_features(s1_r["business_name"], s1_r["business_address"], tr["business_name"], tr["business_address"])
                # Sanity: if name and address both match perfectly, skip (likely true entity duplicate)
                if feats[2] >= 0.85 and feats[9] >= 0.85:
                    continue
                X_train.append(feats)
                y_train.append(0)
                
        # Random Negative
        c_pool = targets_by_country[c]
        rn_id = random.choice(c_pool)
        if rn_id not in targets:
            tr = target_records[rn_id]
            X_train.append(compute_enhanced_features(s1_r["business_name"], s1_r["business_address"], tr["business_name"], tr["business_address"]))
            y_train.append(0)
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"X_train generated in {time.time()-t1:.2f}s: {X_train.shape[0]} pairs (pos={sum(y_train)}, neg={len(y_train)-sum(y_train)})")
    
    print("Fitting HistGradientBoostingClassifier...")
    t2 = time.time()
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=45,
        min_samples_leaf=25,
        early_stopping=True,
        random_state=42
    )
    clf.fit(X_train, y_train)
    print(f"Fit completed in {time.time()-t2:.2f}s (iterations: {clf.n_iter_})")
    
    print("Evaluating on 5,000 Validation Entities with Dual IDF Retrieval...")
    val_pred_scores = defaultdict(list)
    for s1_id, targets in val_gt_dict.items():
        s1_r = s1_val_dict[s1_id]
        c = s1_r["country"]
        
        cand_ids = set()
        if c in indices:
            c_indices = indices[c].query(s1_r["business_name"], s1_r["business_address"], top_name=18, top_addr=12)
            cand_ids = [indices[c].records[i][0] for i in c_indices]
            
        cand_feats = []
        for cid in cand_ids:
            cr = target_records.get(cid)
            if cr:
                cand_feats.append(compute_enhanced_features(s1_r["business_name"], s1_r["business_address"], cr["business_name"], cr["business_address"]))
                
        if cand_feats:
            probs = clf.predict_proba(np.array(cand_feats, dtype=np.float32))[:, 1]
            for cid, p in zip(cand_ids, probs):
                val_pred_scores[s1_id].append((cid, p))
                
    best_th = 0.5
    best_f05 = 0.0
    for th in [0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90]:
        preds = {}
        for s1_id in val_gt_dict:
            matched = set(cid for cid, p in val_pred_scores.get(s1_id, []) if p >= th)
            preds[s1_id] = matched
        score = evaluate_macro_f05(val_gt_dict, preds)
        print(f"  Threshold {th:.2f} -> Validation Macro F0.5: {score:.5f}")
        if score > best_f05:
            best_f05 = score
            best_th = th
            
    print(f"\n>>> BEST RESULT: Threshold {best_th:.2f} with Macro F0.5 = {best_f05:.5f} <<<")

if __name__ == "__main__":
    main()
