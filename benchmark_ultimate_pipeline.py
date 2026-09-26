import sys, os, time, math, re, unicodedata, random
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/code/business_entity_resolution"))

import polars as pl
import numpy as np
from collections import defaultdict
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier

from src.utils import clean_number, LEGAL_STOPWORDS, ADDR_STOPWORDS, evaluate_macro_f05, evaluate_entity_f05

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

BRAHMI_OFFSET_MAP = {
    0x02: "n", 0x03: "h", 0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ee", 0x09: "u", 0x0A: "oo",
    0x0B: "ri", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x12: "o", 0x13: "o", 0x14: "au",
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "nn",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "rr", 0x32: "l", 0x33: "l", 0x34: "lh", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x3E: "aa", 0x3F: "i", 0x40: "ee", 0x41: "u", 0x42: "oo", 0x43: "ri",
    0x46: "e", 0x47: "e", 0x48: "ai", 0x4A: "o", 0x4B: "o", 0x4C: "au",
    0x4D: "",
}
BLOCK_STARTS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]

def transliterate_indic(text):
    if not text: return ""
    res = []
    for ch in text:
        code = ord(ch)
        matched = False
        for b in BLOCK_STARTS:
            if b <= code < b + 0x80:
                offset = code - b
                if offset in BRAHMI_OFFSET_MAP:
                    res.append(BRAHMI_OFFSET_MAP[offset])
                    matched = True
                    break
        if not matched:
            res.append(ch)
    return "".join(res)

def clean_text_fast(text):
    if not text: return ""
    text = transliterate_indic(text)
    text = unicodedata.normalize('NFKD', text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r'\.(com|org|net|co|in|fr|io|info)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_improved_keys(name, address):
    clean_n = clean_text_fast(name)
    clean_a = clean_text_fast(address) if address else ''
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

def compute_ultimate_features(n1, a1, n2, a2, raw_n1, raw_a1, raw_n2, raw_a2):
    # 1. Base name metrics
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
    
    # Pvt vs Public Mismatch feature
    rn1_lower = raw_n1.lower()
    rn2_lower = raw_n2.lower()
    is_pvt1 = ("private" in rn1_lower or "pvt" in rn1_lower)
    is_pub1 = ("public" in rn1_lower)
    is_pvt2 = ("private" in rn2_lower or "pvt" in rn2_lower)
    is_pub2 = ("public" in rn2_lower)
    pvt_pub_mismatch = 1.0 if ((is_pvt1 and is_pub2) or (is_pub1 and is_pvt2)) else 0.0
    
    # 2. Address metrics
    has_a1 = bool(a1)
    has_a2 = bool(a2)
    both_have_addr = 1.0 if (has_a1 and has_a2) else 0.0
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
        exact_name, exact_name_no_legal, first_match, pvt_pub_mismatch,
        addr_null, both_have_addr, a_token_set, a_token_sort, a_jaccard, num_match, bldg_match,
        name_x_addr, max_sim, min_sim
    ]

class FastInvertedIndex:
    def __init__(self, max_key_len=4000):
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
        N = len(self.records)
        for k, posting in self.index.items():
            self.idfs[k] = math.log(1.0 + (N / len(posting)))

    def query(self, name, address, top_name=25, top_addr=15):
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
    print("="*75)
    print("BENCHMARK ULTIMATE PIPELINE (Target: > 0.986 Macro F0.5)")
    print("="*75)
    t0 = time.time()
    
    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
    s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
    
    n_train = 40000
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
            
    s1_train_dict = {r["entity_id"]: r for r in s1_full.filter(pl.col("entity_id").is_in(train_gt["source1_entity_id"].to_list())).iter_rows(named=True)}
    s1_val_dict = {r["entity_id"]: r for r in s1_full.filter(pl.col("entity_id").is_in(val_gt["source1_entity_id"].to_list())).iter_rows(named=True)}
    
    s2_full = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
    s3_full = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")
    
    s2_targets = {x for x in needed_targets if x.startswith("S2-")}
    s3_targets = {x for x in needed_targets if x.startswith("S3-")}
    
    s2_pool = pl.concat([s2_full.filter(pl.col("entity_id").is_in(list(s2_targets))), s2_full.sample(n=100000, seed=42)]).unique(subset=["entity_id"])
    s3_pool = pl.concat([s3_full.filter(pl.col("entity_id").is_in(list(s3_targets))), s3_full.sample(n=100000, seed=42)]).unique(subset=["entity_id"])
    
    target_records = {}
    for r in s2_pool.iter_rows(named=True): target_records[r["entity_id"]] = r
    for r in s3_pool.iter_rows(named=True): target_records[r["entity_id"]] = r
    
    # Pre-clean strings
    s1_clean_train = {r["entity_id"]: (clean_text_fast(r["business_name"]), clean_text_fast(r["business_address"]) if r["business_address"] else "") for r in s1_train_dict.values()}
    s1_clean_val = {r["entity_id"]: (clean_text_fast(r["business_name"]), clean_text_fast(r["business_address"]) if r["business_address"] else "") for r in s1_val_dict.values()}
    target_clean = {r["entity_id"]: (clean_text_fast(r["business_name"]), clean_text_fast(r["business_address"]) if r["business_address"] else "") for r in target_records.values()}
    
    print(f"Loaded datasets in {time.time()-t0:.2f}s")
    
    # Inverted Index per country
    indices = {"US": FastInvertedIndex(max_key_len=4000), "India": FastInvertedIndex(max_key_len=4000)}
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
        print(f"Building FastInvertedIndex for {c} with {len(eids)} records...")
        indices[c].add_records(eids, names, addrs)
        
    print("Generating balanced training pairs with hard negative mining...")
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
                X_train.append(compute_ultimate_features(s1_cn, s1_ca, t_cn, t_ca, s1_r["business_name"], s1_r["business_address"], tr["business_name"], tr["business_address"]))
                y_train.append(1)
                
        # Hard Negatives from blocking index
        if c in indices:
            c_indices = indices[c].query(s1_r["business_name"], s1_r["business_address"], top_name=12, top_addr=8)
            cand_eids = [indices[c].records[i][0] for i in c_indices if indices[c].records[i][0] not in targets]
            for hn_id in cand_eids[:3]:
                tr = target_records[hn_id]
                t_cn, t_ca = target_clean[hn_id]
                feats = compute_ultimate_features(s1_cn, s1_ca, t_cn, t_ca, s1_r["business_name"], s1_r["business_address"], tr["business_name"], tr["business_address"])
                # Skip duplicate ground truth candidates
                if feats[2] >= 0.90 and feats[12] >= 0.90:
                    continue
                X_train.append(feats)
                y_train.append(0)
                
        # Random Negative
        c_pool = targets_by_c[c]
        rn_id = random.choice(c_pool)
        if rn_id not in targets:
            tr = target_records[rn_id]
            t_cn, t_ca = target_clean[rn_id]
            X_train.append(compute_ultimate_features(s1_cn, s1_ca, t_cn, t_ca, s1_r["business_name"], s1_r["business_address"], tr["business_name"], tr["business_address"]))
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
    
    print("Generating Candidates & Predicting on 10,000 Validation Entities...")
    val_cand_pairs = []
    val_cand_features = []
    total_val_targets = sum(len(v) for v in val_gt_dict.values())
    val_hits = 0
    
    for s1_id in val_gt_dict:
        s1_r = s1_val_dict[s1_id]
        c = s1_r["country"]
        gt_targets = val_gt_dict[s1_id]
        s1_cn, s1_ca = s1_clean_val[s1_id]
        
        if c in indices:
            c_indices = indices[c].query(s1_r["business_name"], s1_r["business_address"], top_name=25, top_addr=15)
            cand_eids = [indices[c].records[i][0] for i in c_indices]
            val_hits += len(gt_targets & set(cand_eids))
            
            for cid in cand_eids:
                if cid in target_clean:
                    tr = target_records[cid]
                    t_cn, t_ca = target_clean[cid]
                    feats = compute_ultimate_features(s1_cn, s1_ca, t_cn, t_ca, s1_r["business_name"], s1_r["business_address"], tr["business_name"], tr["business_address"])
                    val_cand_features.append(feats)
                    val_cand_pairs.append((s1_id, cid))
                    
    print(f"Validation Candidate Recall: {val_hits}/{total_val_targets} ({val_hits/total_val_targets*100:.2f}%)")
    print(f"Total Candidate Pairs to score: {len(val_cand_pairs):,} (avg {len(val_cand_pairs)/len(val_gt_dict):.1f} per S1)")
    
    probs = clf.predict_proba(np.array(val_cand_features, dtype=np.float32))[:, 1]
    
    # Evaluate across thresholds with bipartite disambiguation
    print("\n--- Sweeping Decision Threshold with Bipartite Disambiguation ---")
    for th in [0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90]:
        # Filter raw matches above th
        raw_matches = defaultdict(list)
        for (s1_id, cid), p in zip(val_cand_pairs, probs):
            if p >= th:
                raw_matches[s1_id].append((cid, float(p)))
                
        # Winner-takes-all bipartite matching
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
                    
        # Compute Macro F0.5
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
                
        macro_f05 = np.mean(scores)
        mean_p = np.mean(precisions)
        mean_r = np.mean(recalls)
        print(f"Th: {th:.2f} | Macro F0.5: {macro_f05:.6f} | Non-Singleton P: {mean_p:.5f} | R: {mean_r:.5f}")

if __name__ == "__main__":
    main()
