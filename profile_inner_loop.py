import sys, os
sys.path.insert(0, os.path.abspath('code/business_entity_resolution'))

import polars as pl
import time, joblib
from rapidfuzz import fuzz
from src.blocking import InvertedIndex, extract_blocking_keys
from src.utils import clean_text, clean_number
from src.features import compute_features

# Load France sample
s1_fr = pl.read_csv("dataset/test/test_source1.tsv", separator="\t").filter(pl.col("country") == "France").head(500)
s2_fr = pl.read_csv("dataset/test/test_source2.tsv", separator="\t").filter(pl.col("country") == "France").head(50000)
s3_fr = pl.read_csv("dataset/test/test_source3.tsv", separator="\t").filter(pl.col("country") == "France").head(50000)

idx = InvertedIndex(max_key_len=1000)
idx.add_records(s2_fr["entity_id"].to_list(), s2_fr["business_name"].to_list(), s2_fr["business_address"].to_list())
idx.add_records(s3_fr["entity_id"].to_list(), s3_fr["business_name"].to_list(), s3_fr["business_address"].to_list())

artifact = joblib.load("code/business_entity_resolution/src/matching_model.joblib")
clf = artifact["model"]

t_query = 0
t_feats = 0
t_predict = 0

s1_names = s1_fr["business_name"].to_list()
s1_addrs = s1_fr["business_address"].to_list()

for name, addr in zip(s1_names, s1_addrs):
    t0 = time.time()
    cand_indices = idx.query(name, addr, top_k=20)
    t_query += time.time() - t0
    
    if cand_indices:
        t0 = time.time()
        batch_feats = []
        for ci in cand_indices:
            cid, cname, caddr = idx.records[ci]
            batch_feats.append(compute_features(name, addr, cname, caddr))
        t_feats += time.time() - t0
        
        t0 = time.time()
        import numpy as np
        probs = clf.predict_proba(np.array(batch_feats, dtype=np.float32))
        t_predict += time.time() - t0

print(f"500 entities breakdown:")
print(f"  t_query:   {t_query:.3f}s ({t_query/500*1000:.2f} ms/entity)")
print(f"  t_feats:   {t_feats:.3f}s ({t_feats/500*1000:.2f} ms/entity)")
print(f"  t_predict: {t_predict:.3f}s ({t_predict/500*1000:.2f} ms/entity)")
