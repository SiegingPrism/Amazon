import polars as pl

gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
s1_country = dict(zip(
    pl.read_csv("dataset/train/train_source1.tsv", separator="\t", columns=["entity_id", "country"])["entity_id"].to_list(),
    pl.read_csv("dataset/train/train_source1.tsv", separator="\t", columns=["entity_id", "country"])["country"].to_list()
))
s2_country = dict(zip(
    pl.read_csv("dataset/train/train_source2.tsv", separator="\t", columns=["entity_id", "country"])["entity_id"].to_list(),
    pl.read_csv("dataset/train/train_source2.tsv", separator="\t", columns=["entity_id", "country"])["country"].to_list()
))
s3_country = dict(zip(
    pl.read_csv("dataset/train/train_source3.tsv", separator="\t", columns=["entity_id", "country"])["entity_id"].to_list(),
    pl.read_csv("dataset/train/train_source3.tsv", separator="\t", columns=["entity_id", "country"])["country"].to_list()
))

cross_country_count = 0
total_checked = 0
match_counts = []
s2_match_counts = []
s3_match_counts = []

for row in gt.slice(0, 100000).iter_rows(named=True):
    s1_id = row["source1_entity_id"]
    c1 = s1_country[s1_id]
    m_str = row["matched_entity_ids"]
    if not m_str:
        match_counts.append(0)
        continue
    ids = m_str.split(",")
    match_counts.append(len(ids))
    s2_c = sum(1 for x in ids if x.startswith("S2-"))
    s3_c = sum(1 for x in ids if x.startswith("S3-"))
    s2_match_counts.append(s2_c)
    s3_match_counts.append(s3_c)
    for mid in ids:
        total_checked += 1
        c2 = s2_country.get(mid) or s3_country.get(mid)
        if c2 != c1:
            cross_country_count += 1

print(f"Sample 100k S1 entities:")
print(f"Total match pairs checked: {total_checked}")
print(f"Cross country matches: {cross_country_count}")
print(f"Avg matches per S1: {sum(match_counts)/len(match_counts):.2f}")
print(f"Avg S2 matches when matched: {sum(s2_match_counts)/len(s2_match_counts):.2f}")
print(f"Avg S3 matches when matched: {sum(s3_match_counts)/len(s3_match_counts):.2f}")
