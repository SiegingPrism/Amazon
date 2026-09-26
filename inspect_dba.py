import sys
sys.path.insert(0, "code/business_entity_resolution")
import polars as pl
from rapidfuzz import fuzz
from src.utils import clean_text

gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t").slice(0, 50000)
s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
s2_df = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
s3_df = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")

s1_data = {r["entity_id"]: r for r in s1_df.filter(pl.col("entity_id").is_in(gt_df["source1_entity_id"])).iter_rows(named=True)}

needed = set()
for r in gt_df.iter_rows(named=True):
    if r["matched_entity_ids"]:
        needed.update(r["matched_entity_ids"].split(","))

target_data = {}
for df in [s2_df, s3_df]:
    for r in df.filter(pl.col("entity_id").is_in(list(needed))).iter_rows(named=True):
        target_data[r["entity_id"]] = r

count = 0
for r in gt_df.iter_rows(named=True):
    s1_id = r["source1_entity_id"]
    if not r["matched_entity_ids"]: continue
    s1_r = s1_data.get(s1_id, {})
    s1_n = clean_text(s1_r.get("business_name", ""))
    s1_a = clean_text(s1_r.get("business_address", ""))
    for tid in r["matched_entity_ids"].split(","):
        if tid not in target_data: continue
        tr = target_data[tid]
        t_n = clean_text(tr.get("business_name", ""))
        t_a = clean_text(tr.get("business_address", ""))
        ts = fuzz.token_set_ratio(s1_n, t_n) / 100.0
        pr = fuzz.partial_ratio(s1_n, t_n) / 100.0
        fz = fuzz.ratio(s1_n, t_n) / 100.0
        max_n = max(ts, pr, fz)
        if max_n < 0.25:
            ats = fuzz.token_set_ratio(s1_a, t_a) / 100.0 if s1_a and t_a else 0
            has_addr = bool(tr.get("business_address"))
            print("S1:", s1_r.get("business_name"), "|", s1_r.get("business_address"))
            print("TR:", tr.get("business_name"), "|", tr.get("business_address"), f"(a_sim={ats:.2f}, has_addr={has_addr})")
            print("-" * 60)
            count += 1
            if count >= 10: break
    if count >= 10: break
