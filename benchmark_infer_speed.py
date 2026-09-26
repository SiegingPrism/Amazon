import sys, os
sys.path.insert(0, os.path.abspath('code/business_entity_resolution'))

import polars as pl
import time, joblib
from rapidfuzz import fuzz
from src.blocking import InvertedIndex
from src.utils import clean_text, clean_number, LEGAL_STOPWORDS, ADDR_STOPWORDS
from src.features import compute_features

print("Testing end-to-end inference speed on 5,000 test entities...")
model_data = joblib.load("code/business_entity_resolution/src/matching_model.joblib")
clf = model_data["model"]
threshold = model_data["threshold"]
print(f"Loaded model, threshold: {threshold}")

# Test on France test set
s1_fr = pl.read_csv("dataset/test/test_source1.tsv", separator="\t").filter(pl.col("country") == "France").head(5000)
s2_fr = pl.read_csv("dataset/test/test_source2.tsv", separator="\t").filter(pl.col("country") == "France")
s3_fr = pl.read_csv("dataset/test/test_source3.tsv", separator="\t").filter(pl.col("country") == "France")

print(f"France test: {len(s1_fr)} S1, {len(s2_fr)} S2, {len(s3_fr)} S3")
t0 = time.time()
idx = InvertedIndex(max_key_len=1000)
idx.add_records(s2_fr["entity_id"].to_list(), s2_fr["business_name"].to_list(), s2_fr["business_address"].to_list())
idx.add_records(s3_fr["entity_id"].to_list(), s3_fr["business_name"].to_list(), s3_fr["business_address"].to_list())
print(f"Index built in {time.time()-t0:.2f}s ({len(idx.index)} unique keys)")

t1 = time.time()
s1_eids = s1_fr["entity_id"].to_list()
s1_names = s1_fr["business_name"].to_list()
s1_addrs = s1_fr["business_address"].to_list()

total_matches = 0
total_cands = 0
singletons = 0

for s1_id, s1_name, s1_addr in zip(s1_eids, s1_names, s1_addrs):
    cand_indices = idx.query(s1_name, s1_addr, top_k=20)
    cand_ids = [idx.records[i][0] for i in cand_indices]
    total_cands += len(cand_ids)
    
    matched_ids = []
    if cand_indices:
        s1_n_clean = clean_text(s1_name)
        s1_a_clean = clean_text(s1_addr)
        
        batch_feats = []
        batch_cand_ids = []
        for i in cand_indices:
            cid, cname, caddr = idx.records[i]
            cn_clean = clean_text(cname)
            
            # Fast check: exact clean name
            if s1_n_clean and cn_clean and s1_n_clean == cn_clean:
                matched_ids.append(cid)
                continue
                
            batch_feats.append(compute_features(s1_name, s1_addr, cname, caddr))
            batch_cand_ids.append(cid)
            
        if batch_feats:
            probs = clf.predict_proba(batch_feats)[:, 1]
            for cid, p in zip(batch_cand_ids, probs):
                if p >= threshold:
                    matched_ids.append(cid)
                    
    if matched_ids:
        total_matches += len(matched_ids)
    else:
        singletons += 1

el = time.time() - t1
print(f"Processed 5,000 entities in {el:.2f}s ({5000/el:.1f} entities/sec)")
print(f"Avg candidates: {total_cands/5000:.1f}, Avg matches: {total_matches/5000:.2f}, Singletons: {singletons} ({singletons/5000:.2%})")
