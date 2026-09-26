import sys, os
sys.path.insert(0, os.path.abspath('code/business_entity_resolution'))

import polars as pl
import time, joblib
import numpy as np
from src.blocking import InvertedIndex
from src.features import compute_features
from rapidfuzz import fuzz
from src.utils import clean_text

# Load France sample
s1_fr = pl.read_csv("dataset/test/test_source1.tsv", separator="\t").filter(pl.col("country") == "France").head(5000)
s2_fr = pl.read_csv("dataset/test/test_source2.tsv", separator="\t").filter(pl.col("country") == "France").head(50000)
s3_fr = pl.read_csv("dataset/test/test_source3.tsv", separator="\t").filter(pl.col("country") == "France").head(50000)

idx = InvertedIndex(max_key_len=1000)
idx.add_records(s2_fr["entity_id"].to_list(), s2_fr["business_name"].to_list(), s2_fr["business_address"].to_list())
idx.add_records(s3_fr["entity_id"].to_list(), s3_fr["business_name"].to_list(), s3_fr["business_address"].to_list())

artifact = joblib.load("code/business_entity_resolution/src/matching_model.joblib")
clf = artifact["model"]
threshold = artifact["threshold"]

t0 = time.time()
s1_eids = s1_fr["entity_id"].to_list()
s1_names = s1_fr["business_name"].to_list()
s1_addrs = s1_fr["business_address"].to_list()

cand_dict = {}
match_dict = {eid: [] for eid in s1_eids}

batch_feats = []
pair_tracking = [] # (eid, cid)

for eid, name, addr in zip(s1_eids, s1_names, s1_addrs):
    c_indices = idx.query(name, addr, top_k=20)
    c_ids = [idx.records[ci][0] for ci in c_indices]
    cand_dict[eid] = c_ids
    
    n1_clean = clean_text(name)
    for ci in c_indices:
        cid, cname, caddr = idx.records[ci]
        cn_clean = clean_text(cname)
        if fuzz.token_set_ratio(n1_clean, cn_clean) < 30:
            continue
        feats = compute_features(name, addr, cname, caddr)
        batch_feats.append(feats)
        pair_tracking.append((eid, cid))

t1 = time.time()
print(f"Generated {len(batch_feats)} candidate feature pairs in {t1-t0:.2f}s")

if batch_feats:
    t_pred = time.time()
    X = np.array(batch_feats, dtype=np.float32)
    probs = clf.predict_proba(X)[:, 1]
    print(f"Batched predict_proba on {len(X)} rows in {time.time()-t_pred:.3f}s!")
    
    for (eid, cid), p in zip(pair_tracking, probs):
        if p >= threshold:
            match_dict[eid].append(cid)

total_el = time.time() - t0
print(f"Total time for 5,000 entities: {total_el:.2f}s -> {5000/total_el:.0f} entities/sec!")
