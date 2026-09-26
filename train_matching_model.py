import polars as pl
import numpy as np
import re, unicodedata, time
from collections import defaultdict
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier
import random

print("Loading training ground truth sample for modeling...")
gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")

# Split 10,000 train S1, 5,000 val S1
train_gt = gt_df.slice(0, 10000)
val_gt = gt_df.slice(10000, 5000)

s1_train_dict = {r["entity_id"]: r for r in s1_full.filter(pl.col("entity_id").is_in(train_gt["source1_entity_id"].to_list())).iter_rows(named=True)}
s1_val_dict = {r["entity_id"]: r for r in s1_full.filter(pl.col("entity_id").is_in(val_gt["source1_entity_id"].to_list())).iter_rows(named=True)}

needed_targets = set()
train_gt_dict = {}
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

s2_needed = {x for x in needed_targets if x.startswith("S2-")}
s3_needed = {x for x in needed_targets if x.startswith("S3-")}

s2_full = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
s3_full = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")

target_records = {}
for r in s2_full.filter(pl.col("entity_id").is_in(list(s2_needed))).iter_rows(named=True):
    target_records[r["entity_id"]] = r
for r in s3_full.filter(pl.col("entity_id").is_in(list(s3_needed))).iter_rows(named=True):
    target_records[r["entity_id"]] = r

print(f"Loaded {len(s1_train_dict)} train S1, {len(s1_val_dict)} val S1, {len(target_records)} target S2/S3 records.")

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

def clean_text(text):
    if not text: return ""
    text = transliterate_indic(text)
    text = unicodedata.normalize('NFKD', text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r'\.(com|org|net|co|in|fr|io|info)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

LEGAL_STOPWORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'llp', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'services', 'service', 'enterprises',
    'group', 'holdings', 'holding', 'associates', 'the', 'and', 'of', 'in', 'at',
    'sarl', 'sas', 'sasu', 'eurl', 'sci', 'sa', 'foundation', 'club', 'centre',
    'center', 'international', 'global', 'solutions', 'technologies', 'technology',
    'to', 'for', 'by', 'on', 'with', 'an', 'a', 'de', 'du', 'des', 'la', 'le', 'les',
    'mr', 'mrs', 'ms', 'm', 'smt'
}

ADDR_STOPWORDS = {
    'street', 'st', 'road', 'rd', 'avenue', 'ave', 'drive', 'dr', 'lane', 'ln',
    'boulevard', 'blvd', 'way', 'court', 'ct', 'place', 'pl', 'highway', 'hwy',
    'floor', 'fl', 'suite', 'ste', 'room', 'rm', 'building', 'bldg', 'near',
    'opp', 'opposite', 'behind', 'post', 'box', 'po', 'rue', 'r', 'bd', 'av',
    'allée', 'allee', 'impasse', 'chemin', 'ch', 'residence', 'res', 'appartement',
    'apt', 'north', 'south', 'east', 'west', 'n', 's', 'e', 'w', 'null', 'none'
}

def clean_number(w):
    digits = re.sub(r'\D', '', w)
    return digits.lstrip('0') if digits else ""

def compute_features(s1_name, s1_addr, s2_name, s2_addr):
    n1_clean = clean_text(s1_name)
    n2_clean = clean_text(s2_name)
    
    n_fuzz_ratio = fuzz.ratio(n1_clean, n2_clean) / 100.0
    n_token_sort = fuzz.token_sort_ratio(n1_clean, n2_clean) / 100.0
    n_token_set = fuzz.token_set_ratio(n1_clean, n2_clean) / 100.0
    n_partial = fuzz.partial_ratio(n1_clean, n2_clean) / 100.0
    
    w1 = set(w for w in n1_clean.split() if w not in LEGAL_STOPWORDS and len(w) >= 3)
    w2 = set(w for w in n2_clean.split() if w not in LEGAL_STOPWORDS and len(w) >= 3)
    n_jaccard = len(w1 & w2) / max(1, len(w1 | w2))
    n_overlap = len(w1 & w2)
    
    addr_null = 1.0 if not s2_addr else 0.0
    if not s2_addr:
        a_token_set = 0.0
        a_token_sort = 0.0
        a_jaccard = 0.0
        num_match = 0.5
    else:
        a1_clean = clean_text(s1_addr)
        a2_clean = clean_text(s2_addr)
        a_token_set = fuzz.token_set_ratio(a1_clean, a2_clean) / 100.0
        a_token_sort = fuzz.token_sort_ratio(a1_clean, a2_clean) / 100.0
        
        aw1 = set(w for w in a1_clean.split() if w not in ADDR_STOPWORDS and len(w) >= 3)
        aw2 = set(w for w in a2_clean.split() if w not in ADDR_STOPWORDS and len(w) >= 3)
        a_jaccard = len(aw1 & aw2) / max(1, len(aw1 | aw2))
        
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

def evaluate_macro_f05(gt_dict, pred_dict):
    scores = []
    for s1_id, gt_set in gt_dict.items():
        pred_set = pred_dict.get(s1_id, set())
        if len(gt_set) == 0:
            scores.append(1.0 if len(pred_set) == 0 else 0.0)
            continue
        if len(pred_set) == 0:
            scores.append(0.0)
            continue
        tp = len(gt_set & pred_set)
        if tp == 0:
            scores.append(0.0)
            continue
        p = tp / len(pred_set)
        r = tp / len(gt_set)
        denom = 0.25 * p + r
        scores.append((1.25 * p * r) / denom if denom > 0 else 0.0)
    return np.mean(scores)

print("Building training pairs...")
X_train = []
y_train = []

# Positive pairs
for s1_id, targets in train_gt_dict.items():
    s1_r = s1_train_dict[s1_id]
    for tid in targets:
        tr = target_records.get(tid)
        if tr:
            feats = compute_features(s1_r["business_name"], s1_r["business_address"], tr["business_name"], tr["business_address"])
            X_train.append(feats)
            y_train.append(1)

# Negative pairs
random.seed(42)
targets_by_country = defaultdict(list)
for tid, tr in target_records.items():
    targets_by_country[tr["country"]].append(tid)

for s1_id, targets in train_gt_dict.items():
    s1_r = s1_train_dict[s1_id]
    c = s1_r["country"]
    c_pool = targets_by_country[c]
    negs = random.sample(c_pool, min(4, len(c_pool)))
    for neg_id in negs:
        if neg_id not in targets:
            tr = target_records[neg_id]
            feats = compute_features(s1_r["business_name"], s1_r["business_address"], tr["business_name"], tr["business_address"])
            X_train.append(feats)
            y_train.append(0)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
print(f"X_train shape: {X_train.shape}, Pos: {sum(y_train)}, Neg: {len(y_train)-sum(y_train)}")

print("Training HistGradientBoostingClassifier...")
clf = HistGradientBoostingClassifier(max_iter=150, min_samples_leaf=20, random_state=42)
clf.fit(X_train, y_train)

# Evaluate on Validation Set
print("Evaluating on 5,000 Validation Entities...")
# For each validation S1, test candidate targets from its country pool (true targets + 5 random negatives)
val_pred_scores = defaultdict(list)
for s1_id, targets in val_gt_dict.items():
    s1_r = s1_val_dict[s1_id]
    c = s1_r["country"]
    c_pool = targets_by_country[c]
    test_cands = set(targets)
    # add some negatives
    test_cands.update(random.sample(c_pool, min(5, len(c_pool))))
    
    cand_feats = []
    cand_ids = []
    for cid in test_cands:
        cr = target_records.get(cid)
        if cr:
            cand_feats.append(compute_features(s1_r["business_name"], s1_r["business_address"], cr["business_name"], cr["business_address"]))
            cand_ids.append(cid)
            
    if cand_feats:
        probs = clf.predict_proba(np.array(cand_feats, dtype=np.float32))[:, 1]
        for cid, p in zip(cand_ids, probs):
            val_pred_scores[s1_id].append((cid, p))

# Find optimal threshold for F_0.5
best_th = 0.5
best_f05 = 0.0

for th in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
    preds = {}
    for s1_id in val_gt_dict:
        matched = set(cid for cid, p in val_pred_scores.get(s1_id, []) if p >= th)
        preds[s1_id] = matched
    score = evaluate_macro_f05(val_gt_dict, preds)
    print(f"Threshold: {th:.2f} -> Macro F_0.5: {score:.4f}")
    if score > best_f05:
        best_f05 = score
        best_th = th

print(f"\nBest Validation Macro F_0.5: {best_f05:.4f} at threshold {best_th:.2f}")
