import polars as pl
import re, unicodedata
from collections import defaultdict

# Load 1,000 S1 sample
gt_sample = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t").sample(n=1000, seed=42)
s1_needed = set(gt_sample["source1_entity_id"].to_list())
targets_needed = set()
gt_dict = {}
for r in gt_sample.iter_rows(named=True):
    s1_id = r["source1_entity_id"]
    if r["matched_entity_ids"]:
        m = set(r["matched_entity_ids"].split(","))
        gt_dict[s1_id] = m
        targets_needed.update(m)
    else:
        gt_dict[s1_id] = set()

s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(s1_needed)))
s2_df = pl.read_csv("dataset/train/train_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(targets_needed)))
s3_df = pl.read_csv("dataset/train/train_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(targets_needed)))

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
    # Strip domain extensions
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
    # Extract digits and strip leading zeros
    digits = re.sub(r'\D', '', w)
    return digits.lstrip('0') if digits else ""

def extract_keys(name, address):
    keys = []
    clean_n = clean_text(name)
    words = [w for w in clean_n.split() if w not in LEGAL_STOPWORDS and len(w) >= 3]
    
    # 1. Full clean name (first 3 words joined)
    if words:
        keys.append("NF:" + "_".join(words[:3]))
        for w in words[:4]:
            keys.append("NW:" + w)
            # Add 4-prefix if long word
            if len(w) >= 6:
                keys.append("NP:" + w[:5])
    
    # 2. Address keys
    if address:
        clean_a = clean_text(address)
        raw_words = clean_a.split()
        nums = [clean_number(w) for w in raw_words if any(c.isdigit() for c in w)]
        nums = [n for n in nums if n] # non-empty
        awords = [w for w in raw_words if w not in ADDR_STOPWORDS and not any(c.isdigit() for c in w) and len(w) >= 3]
        
        # Compound key: number + significant address words
        if nums and awords:
            for num in nums[:2]:
                for aw in awords[:4]:
                    keys.append(f"ANW:{num}_{aw}")
                    
        # Word pair inside address (e.g. street + city)
        if len(awords) >= 2:
            keys.append(f"AAW:{awords[0]}_{awords[1]}")
            if len(awords) >= 3:
                keys.append(f"AAW:{awords[0]}_{awords[-1]}")
                
        # First name word + address word
        if words and awords:
            keys.append(f"NAW:{words[0]}_{awords[0]}")
            
    return keys

# Check overlap
missed_count = 0
total_pairs = 0
examples = []

target_map = {}
for df in [s2_df, s3_df]:
    for r in df.iter_rows(named=True):
        target_map[r["entity_id"]] = r

for r1 in s1_df.iter_rows(named=True):
    s1_id = r1["entity_id"]
    s1_keys = set(extract_keys(r1["business_name"], r1["business_address"]))
    targets = gt_dict[s1_id]
    for tid in targets:
        total_pairs += 1
        r2 = target_map.get(tid)
        if not r2: continue
        t_keys = set(extract_keys(r2["business_name"], r2["business_address"]))
        if not (s1_keys & t_keys):
            missed_count += 1
            if len(examples) < 10:
                examples.append((r1, r2, s1_keys, t_keys))

print(f"Total true pairs: {total_pairs}, Missed: {missed_count} ({missed_count/total_pairs:.2%})")
print(f"Candidate Pair Recall: {1.0 - (missed_count / total_pairs):.2%}")
