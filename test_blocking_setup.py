import polars as pl
import re, unicodedata
from collections import defaultdict, Counter
import time

print("Loading train sample...")
t0 = time.time()
gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
# Sample 20,000 S1 entities
val_gt = gt.sample(n=20000, seed=123)
val_s1_ids = set(val_gt["source1_entity_id"].to_list())

val_gt_dict = {}
all_true_targets = set()
for r in val_gt.iter_rows(named=True):
    s1_id = r["source1_entity_id"]
    if r["matched_entity_ids"]:
        targets = set(r["matched_entity_ids"].split(","))
        val_gt_dict[s1_id] = targets
        all_true_targets.update(targets)
    else:
        val_gt_dict[s1_id] = set()

# Load S1 records
s1 = pl.read_csv("dataset/train/train_source1.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(val_s1_ids)))

# We need S2 and S3 records for evaluation:
# To measure realistic blocking, we should search against a substantial pool of S2 and S3 (e.g. 500,000 rows each)
# containing all true targets + background noise!
print(f"Loaded {len(s1)} validation S1 entities with {len(all_true_targets)} true matches.")

# Let's inspect true matches count
total_true_matches = sum(len(v) for v in val_gt_dict.values())
singletons = sum(1 for v in val_gt_dict.values() if len(v) == 0)
print(f"Total true matches: {total_true_matches}, Singletons: {singletons} ({singletons/len(val_gt_dict):.2%})")
