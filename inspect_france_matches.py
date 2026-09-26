import polars as pl
import joblib, os, sys
sys.path.insert(0, os.path.abspath('code/business_entity_resolution'))

from src.blocking import InvertedIndex
from src.utils import clean_text
from src.features import compute_features

model_data = joblib.load("code/business_entity_resolution/src/matching_model.joblib")
clf = model_data["model"]

s1_fr = pl.read_csv("dataset/test/test_source1.tsv", separator="\t").filter(pl.col("country") == "France").head(10)
s2_fr = pl.read_csv("dataset/test/test_source2.tsv", separator="\t").filter(pl.col("country") == "France")
s3_fr = pl.read_csv("dataset/test/test_source3.tsv", separator="\t").filter(pl.col("country") == "France")

idx = InvertedIndex(max_key_len=1000)
idx.add_records(s2_fr["entity_id"].to_list(), s2_fr["business_name"].to_list(), s2_fr["business_address"].to_list())
idx.add_records(s3_fr["entity_id"].to_list(), s3_fr["business_name"].to_list(), s3_fr["business_address"].to_list())

for r in s1_fr.iter_rows(named=True):
    s1_id = r["entity_id"]
    s1_name = r["business_name"]
    s1_addr = r["business_address"]
    s1_n_clean = clean_text(s1_name)
    
    cand_indices = idx.query(s1_name, s1_addr, top_k=20)
    print("\n" + "="*50)
    print("S1:", s1_id, "|", s1_name, "|", s1_addr)
    print("S1 clean name:", s1_n_clean)
    print(f"Top candidates ({len(cand_indices)}):")
    
    for i in cand_indices[:8]:
        cid, cname, caddr = idx.records[i]
        cn_clean = clean_text(cname)
        feats = compute_features(s1_name, s1_addr, cname, caddr)
        prob = clf.predict_proba([feats])[0, 1]
        print(f"  -> {cid} | Prob: {prob:.4f} | Name: {cname} | Addr: {caddr}")
        print(f"     Clean: {cn_clean} | ExactNameMatch: {s1_n_clean == cn_clean}")
