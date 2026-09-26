import polars as pl

# Read small sample of ground truth with non-empty matches
gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
print("Total GT rows:", len(gt))
null_matches = gt.filter(pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids") == ""))
print("Singletons (0 matches):", len(null_matches), f"({len(null_matches)/len(gt):.2%})")

# Sample some rows with matches and some singletons
sample_gt = gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != "")).head(10)
s1_ids = sample_gt["source1_entity_id"].to_list()

s1 = pl.read_csv("dataset/train/train_source1.tsv", separator="\t").filter(pl.col("entity_id").is_in(s1_ids))
s2 = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
s3 = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")

for row in sample_gt.iter_rows(named=True):
    s1_id = row["source1_entity_id"]
    matches = row["matched_entity_ids"].split(",")
    s1_row = s1.filter(pl.col("entity_id") == s1_id).to_dicts()
    if not s1_row:
        continue
    print(f"\n==========================================")
    print(f"S1: {s1_id} | Country: {s1_row[0]['country']}")
    print(f"    Name:    {s1_row[0]['business_name']}")
    print(f"    Address: {s1_row[0]['business_address']}")
    print("MATCHES:")
    for m in matches:
        if m.startswith("S2-"):
            m_row = s2.filter(pl.col("entity_id") == m).to_dicts()
        else:
            m_row = s3.filter(pl.col("entity_id") == m).to_dicts()
        if m_row:
            print(f"  -> {m} | Country: {m_row[0]['country']}")
            print(f"     Name:    {m_row[0]['business_name']}")
            print(f"     Address: {m_row[0]['business_address']}")
