import polars as pl
import re, unicodedata, time
from collections import defaultdict

t0 = time.time()
print("Loading data...")
s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
s2_df = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
s3_df = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")
gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")

# Sample 10k S1 entities for quick evaluation
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

print(f"Validation: 10,000 S1 records, {total_gt_matches} true matches.")

# Filter s1 validation records
val_s1_records = s1_df.filter(pl.col("entity_id").is_in(list(val_s1_ids))).to_dicts()

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
    0x4D: "", # virama
}
BLOCK_STARTS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]

def transliterate_indic(text):
    if not text:
        return ""
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
    if not text:
        return ""
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

def extract_blocking_keys(name, address):
    keys = []
    clean_n = clean_text(name)
    name_words = [w for w in clean_n.split() if w not in LEGAL_STOPWORDS and len(w) >= 3]
    
    if name_words:
        keys.append("N_FULL:" + "_".join(name_words[:3]))
        for w in name_words[:4]:
            keys.append("N_W:" + w)
    
    if address:
        clean_a = clean_text(address)
        # Extract digits/numbers
        nums = [w for w in clean_a.split() if any(c.isdigit() for c in w)]
        addr_words = [w for w in clean_a.split() if w not in ADDR_STOPWORDS and not any(c.isdigit() for c in w) and len(w) >= 3]
        
        if nums and addr_words:
            first_num = nums[0]
            for aw in addr_words[:2]:
                keys.append(f"A_NW:{first_num}_{aw}")
        
        if name_words and addr_words:
            keys.append(f"NA:{name_words[0]}_{addr_words[0]}")
            
    return keys

print("Building index on train S2 + S3 for benchmark...")
t1 = time.time()
# Let's index S2 and S3 partitioned by country
# Since validation has US and India, we can test on US and India separately
index_by_country = {
    "US": defaultdict(list),
    "India": defaultdict(list)
}

# Index S2
for row in s2_df.iter_rows(named=True):
    c = row["country"]
    if c in index_by_country:
        eid = row["entity_id"]
        keys = extract_blocking_keys(row["business_name"], row["business_address"])
        for k in set(keys):
            index_by_country[c][k].append(eid)

# Index S3
for row in s3_df.iter_rows(named=True):
    c = row["country"]
    if c in index_by_country:
        eid = row["entity_id"]
        keys = extract_blocking_keys(row["business_name"], row["business_address"])
        for k in set(keys):
            index_by_country[c][k].append(eid)

print(f"Indexing completed in {time.time()-t1:.2f}s")
for c in ["US", "India"]:
    print(f"Country {c}: {len(index_by_country[c])} unique blocking keys.")

# Query validation set
print("\nQuerying candidates for 10k validation entities...")
t2 = time.time()
MAX_KEY_FREQ = 1000 # Ignore keys with > 1000 matches to prevent huge candidate explosion

total_candidates = 0
found_matches = 0
candidates_per_entity = []

for r in val_s1_records:
    s1_id = r["entity_id"]
    c = r["country"]
    keys = extract_blocking_keys(r["business_name"], r["business_address"])
    
    cand_set = set()
    idx = index_by_country[c]
    for k in keys:
        posting = idx.get(k)
        if posting and len(posting) <= MAX_KEY_FREQ:
            cand_set.update(posting)
    
    total_candidates += len(cand_set)
    candidates_per_entity.append(len(cand_set))
    
    true_targets = gt_dict[s1_id]
    if true_targets:
        hits = len(cand_set & true_targets)
        found_matches += hits

print(f"Querying completed in {time.time()-t2:.2f}s")
print(f"Recall: {found_matches} / {total_gt_matches} = {found_matches / total_gt_matches:.2%}")
print(f"Avg candidates per S1 entity: {total_candidates / len(val_s1_records):.1f}")
print(f"Median candidates per S1: {sorted(candidates_per_entity)[len(candidates_per_entity)//2]}")
