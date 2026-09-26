import polars as pl
import re
from collections import Counter
import unicodedata

# Read sample of ground truth with matches
gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
matched_gt = gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != "")).sample(n=20000, seed=42)

s1_sample_ids = set(matched_gt["source1_entity_id"].to_list())
s1 = pl.read_csv("dataset/train/train_source1.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(s1_sample_ids)))

s2_needed = set()
s3_needed = set()
for r in matched_gt.iter_rows(named=True):
    for m in r["matched_entity_ids"].split(","):
        if m.startswith("S2-"):
            s2_needed.add(m)
        else:
            s3_needed.add(m)

print(f"Sampled 20k S1 entities with {len(s2_needed)} S2 targets and {len(s3_needed)} S3 targets.")

s2 = pl.read_csv("dataset/train/train_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(s2_needed)))
s3 = pl.read_csv("dataset/train/train_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(s3_needed)))

s1_dict = {r["entity_id"]: r for r in s1.iter_rows(named=True)}
s2_dict = {r["entity_id"]: r for r in s2.iter_rows(named=True)}
s3_dict = {r["entity_id"]: r for r in s3.iter_rows(named=True)}

def normalize_text(text):
    if not text:
        return ""
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
    'center', 'international', 'global', 'solutions', 'technologies', 'technology'
}

def get_tokens(text, remove_legal=True):
    norm = normalize_text(text)
    tokens = set(norm.split())
    if remove_legal:
        tokens = {t for t in tokens if t not in LEGAL_STOPWORDS and len(t) > 1}
    return tokens

# Check token overlaps in ground truth pairs
total_pairs = 0
name_overlap_count = 0
name_exact_count = 0
addr_overlap_count = 0
addr_null_count = 0
name_or_addr_overlap = 0

diff_country = 0

for r in matched_gt.iter_rows(named=True):
    s1_id = r["source1_entity_id"]
    r1 = s1_dict[s1_id]
    s1_name_toks = get_tokens(r1["business_name"])
    s1_addr_toks = get_tokens(r1["business_address"], remove_legal=False)
    s1_name_norm = normalize_text(r1["business_name"])

    for mid in r["matched_entity_ids"].split(","):
        r2 = s2_dict.get(mid) or s3_dict.get(mid)
        if not r2:
            continue
        total_pairs += 1
        if r1["country"] != r2["country"]:
            diff_country += 1

        r2_name_toks = get_tokens(r2["business_name"])
        r2_addr_toks = get_tokens(r2["business_address"], remove_legal=False) if r2["business_address"] else set()
        r2_name_norm = normalize_text(r2["business_name"])

        has_name_overlap = len(s1_name_toks & r2_name_toks) > 0
        has_addr_overlap = len(s1_addr_toks & r2_addr_toks) > 0
        is_name_exact = (s1_name_norm == r2_name_norm)

        if has_name_overlap:
            name_overlap_count += 1
        if is_name_exact:
            name_exact_count += 1
        if has_addr_overlap:
            addr_overlap_count += 1
        if not r2["business_address"]:
            addr_null_count += 1
        if has_name_overlap or has_addr_overlap:
            name_or_addr_overlap += 1

print(f"\n--- Ground Truth Analysis ({total_pairs} pairs) ---")
print(f"Diff country: {diff_country}")
print(f"Name significant token overlap: {name_overlap_count} ({name_overlap_count/total_pairs:.2%})")
print(f"Name exact normalized match:    {name_exact_count} ({name_exact_count/total_pairs:.2%})")
print(f"Address token overlap:          {addr_overlap_count} ({addr_overlap_count/total_pairs:.2%})")
print(f"Address null in S2/S3:          {addr_null_count} ({addr_null_count/total_pairs:.2%})")
print(f"Name OR Address overlap:        {name_or_addr_overlap} ({name_or_addr_overlap/total_pairs:.2%})")
