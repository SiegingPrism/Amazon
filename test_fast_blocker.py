import polars as pl
import re, unicodedata, time
from collections import defaultdict

t0 = time.time()
print("Loading train data...")
s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
s2_df = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
s3_df = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")
gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")

# Sample 10k S1 entities for validation
val_sample = gt_df.sample(n=10000, seed=42)
val_s1_ids = set(val_sample["source1_entity_id"].to_list())

gt_dict = {}
total_gt_matches = 0
for r in val_sample.iter_rows(named=True):
    s1_id = r["source1_entity_id"]
    if r["matched_entity_ids"]:
        targets = set(r["matched_entity_ids"].split(","))
        gt_dict[s1_id] = targets
        total_gt_matches += len(targets)
    else:
        gt_dict[s1_id] = set()

# Filter val S1
val_s1 = s1_df.filter(pl.col("entity_id").is_in(list(val_s1_ids)))
val_eids = val_s1["entity_id"].to_list()
val_names = val_s1["business_name"].to_list()
val_addrs = val_s1["business_address"].to_list()
val_countries = val_s1["country"].to_list()

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
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

LEGAL_STOPWORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'llp', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'services', 'service', 'enterprises',
    'group', 'holdings', 'holding', 'associates', 'the', 'and', 'of', 'in', 'at',
    'sarl', 'sas', 'sasu', 'eurl', 'sci', 'sa', 'foundation', 'club', 'centre',
    'center', 'international', 'global', 'solutions', 'technologies', 'technology',
    'to', 'for', 'by', 'on', 'with', 'an', 'a', 'de', 'du', 'des', 'la', 'le', 'les'
}

ADDR_STOPWORDS = {
    'street', 'st', 'road', 'rd', 'avenue', 'ave', 'drive', 'dr', 'lane', 'ln',
    'boulevard', 'blvd', 'way', 'court', 'ct', 'place', 'pl', 'highway', 'hwy',
    'floor', 'fl', 'suite', 'ste', 'room', 'rm', 'building', 'bldg', 'near',
    'opp', 'opposite', 'behind', 'post', 'box', 'po', 'rue', 'r', 'bd', 'av',
    'allée', 'allee', 'impasse', 'chemin', 'ch', 'residence', 'res', 'appartement',
    'apt', 'north', 'south', 'east', 'west', 'n', 's', 'e', 'w', 'null', 'none'
}

def extract_keys(name, address):
    keys = []
    clean_n = clean_text(name)
    words = [w for w in clean_n.split() if w not in LEGAL_STOPWORDS and len(w) >= 3]
    
    # 1. Full clean name (first 3 words joined)
    if words:
        keys.append("NF:" + "_".join(words[:3]))
        for w in words[:3]:
            keys.append("NW:" + w)
    
    # 2. Address keys
    if address:
        clean_a = clean_text(address)
        nums = [w for w in clean_a.split() if any(c.isdigit() for c in w)]
        awords = [w for w in clean_a.split() if w not in ADDR_STOPWORDS and not any(c.isdigit() for c in w) and len(w) >= 3]
        
        if nums and awords:
            num = nums[0]
            for aw in awords[:2]:
                keys.append(f"ANW:{num}_{aw}")
                
        if words and awords:
            keys.append(f"NAW:{words[0]}_{awords[0]}")
            
    return keys

print("Building country index on train S2 + S3...")
t1 = time.time()
index = {"US": defaultdict(list), "India": defaultdict(list)}

for df, src in [(s2_df, "S2"), (s3_df, "S3")]:
    eids = df["entity_id"].to_list()
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    countries = df["country"].to_list()
    for eid, name, addr, country in zip(eids, names, addrs, countries):
        if country in index:
            c_idx = index[country]
            for k in set(extract_keys(name, addr)):
                c_idx[k].append(eid)

print(f"Index built in {time.time()-t1:.2f}s")
for c in ["US", "India"]:
    print(f"Country {c}: {len(index[c])} unique keys.")

# Query validation
t2 = time.time()
MAX_KEY_LEN = 1000

total_cands = 0
found_matches = 0
cand_counts = []

for eid, name, addr, country in zip(val_eids, val_names, val_addrs, val_countries):
    c_idx = index[country]
    cands = set()
    for k in extract_keys(name, addr):
        posting = c_idx.get(k)
        if posting and len(posting) <= MAX_KEY_LEN:
            cands.update(posting)
    
    total_cands += len(cands)
    cand_counts.append(len(cands))
    
    targets = gt_dict[eid]
    if targets:
        found_matches += len(cands & targets)

print(f"Queried 10k entities in {time.time()-t2:.2f}s")
print(f"Candidate Recall: {found_matches} / {total_gt_matches} = {found_matches/total_gt_matches:.2%}")
print(f"Avg candidates per S1: {total_cands/len(val_eids):.1f}")
print(f"Median candidates: {sorted(cand_counts)[len(cand_counts)//2]}")
