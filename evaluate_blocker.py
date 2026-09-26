import polars as pl
import re, unicodedata, time
from collections import defaultdict

t0 = time.time()
print("Loading train data...")
s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
s2_full = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
s3_full = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")
gt_full = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")

print(f"Data loaded in {time.time()-t0:.2f}s")

# Let's test on India first, or US first, or both
# Let's take 10,000 S1 entities from India and 10,000 S1 from US
val_sample = gt_full.sample(n=20000, seed=42)
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

print(f"Validation sample: 20k S1 entities, {total_gt_matches} total true matches.")

# Indic transliterator
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
    # Transliterate Indic scripts first if present
    text = transliterate_indic(text)
    # Strip accents
    text = unicodedata.normalize('NFKD', text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

COMMON_LEGAL = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'llp', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'sarl', 'sas', 'sasu', 'eurl', 'sci', 'sa',
    'the', 'and', 'of', 'in', 'at', 'to', 'for', 'by', 'on', 'with', 'an', 'a'
}

print("Testing text cleaner...")
print("Raw: 'राम मार्केटिंग प्राइवेट लिमिटेड' -> Cleaned:", clean_text("राम मार्केटिंग प्राइवेट लिमिटेड"))
print("Raw: '<< Team Ecole' -> Cleaned:", clean_text("<< Team Ecole"))
print("Raw: '5 bis Rue Pierre Dignac, Bordeaux' -> Cleaned:", clean_text("5 bis Rue Pierre Dignac, Bordeaux"))
