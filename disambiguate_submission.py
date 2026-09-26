import polars as pl
import joblib, time, os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/code/business_entity_resolution"))

import numpy as np
from collections import defaultdict
from src.features import compute_features_precleaned
from src.utils import clean_text

def main():
    print("="*60)
    print("Competitive Target Disambiguation (Bipartite Optimization)")
    print("="*60)
    t0 = time.time()
    
    match_path = "output/matching_results.tsv"
    cand_path = "output/candidate_pairs.tsv"
    
    # 1. Read test_source1 to preserve exact original entity order
    s1_full = pl.read_csv("dataset/test/test_source1.tsv", separator="\t")
    original_s1_ids = s1_full["entity_id"].to_list()
    n_s1 = len(original_s1_ids)
    
    mr = pl.read_csv(match_path, separator="\t")
    
    # 2. Map target_id -> list of s1_ids
    target_to_s1 = defaultdict(list)
    for r in mr.iter_rows(named=True):
        m = r["matched_entity_ids"]
        if m:
            for tid in m.split(","):
                target_to_s1[tid].append(r["source1_entity_id"])
                
    duplicates = {tid: s1_list for tid, s1_list in target_to_s1.items() if len(s1_list) > 1}
    print(f"Total target predictions: {sum(len(v) for v in target_to_s1.values()):,}")
    print(f"Unique target records:    {len(target_to_s1):,}")
    print(f"Conflicting target IDs:   {len(duplicates):,} across {sum(len(v) for v in duplicates.values()):,} assignments")
    
    needed_s1 = set(s1_id for s1_list in duplicates.values() for s1_id in s1_list)
    needed_targets = set(duplicates.keys())
    
    print(f"Loading {len(needed_s1):,} S1 and {len(needed_targets):,} target records for re-scoring...")
    t1 = time.time()
    s1_test = pl.read_csv("dataset/test/test_source1.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(needed_s1)))
    s2_test = pl.read_csv("dataset/test/test_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(needed_targets)))
    s3_test = pl.read_csv("dataset/test/test_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(needed_targets)))
    print(f"Loaded test records in {time.time()-t1:.2f}s")
    
    t2 = time.time()
    s1_clean = {}
    for r in s1_test.iter_rows(named=True):
        s1_clean[r["entity_id"]] = (clean_text(r["business_name"]), clean_text(r["business_address"]) if r["business_address"] else "")
        
    target_clean = {}
    for df in [s2_test, s3_test]:
        for r in df.iter_rows(named=True):
            target_clean[r["entity_id"]] = (clean_text(r["business_name"]), clean_text(r["business_address"]) if r["business_address"] else "", bool(r["business_address"]))
    print(f"Pre-cleaned text in {time.time()-t2:.2f}s")
    
    # Load matching model
    artifact = joblib.load("code/business_entity_resolution/src/matching_model.joblib")
    clf = artifact["model"]
    
    t3 = time.time()
    pairs_to_score = []
    feats_to_score = []
    for tid, s1_list in duplicates.items():
        t_cn, t_ca, t_ha = target_clean[tid]
        for s1_id in s1_list:
            s1_cn, s1_ca = s1_clean[s1_id]
            pairs_to_score.append((tid, s1_id))
            feats_to_score.append(compute_features_precleaned(s1_cn, s1_ca, t_cn, t_ca, t_ha))
            
    print(f"Extracted {len(feats_to_score):,} feature vectors in {time.time()-t3:.2f}s")
    
    t4 = time.time()
    probs = clf.predict_proba(np.array(feats_to_score, dtype=np.float32))[:, 1]
    print(f"Computed model probabilities in {time.time()-t4:.2f}s")
    
    # Disambiguate: assign each target ID to the single S1 entity with highest probability
    target_winner = {}
    for (tid, s1_id), p in zip(pairs_to_score, probs):
        if tid not in target_winner or p > target_winner[tid][1]:
            target_winner[tid] = (s1_id, p)
            
    # Reconstruct clean match_results dictionary
    clean_matches = defaultdict(list)
    for r in mr.iter_rows(named=True):
        s1_id = r["source1_entity_id"]
        m = r["matched_entity_ids"]
        if m:
            for tid in m.split(","):
                if tid in target_winner:
                    if s1_id == target_winner[tid][0]:
                        clean_matches[s1_id].append(tid)
                else:
                    clean_matches[s1_id].append(tid)
                    
    # Write updated matching_results.tsv in exact original order
    t5 = time.time()
    with open(match_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in original_s1_ids:
            m_list = clean_matches.get(s1_id, [])
            f_out.write(f"{s1_id}\t{','.join(m_list)}\n")
            
    print(f"Cleaned matching_results.tsv written in {time.time()-t5:.2f}s")
    
    # Final stats
    total_clean_matches = sum(len(v) for v in clean_matches.values())
    clean_singletons = sum(1 for s1_id in original_s1_ids if len(clean_matches[s1_id]) == 0)
    print("\n" + "="*60)
    print("DISAMBIGUATION COMPLETE!")
    print(f"Total S1 Entities:     {n_s1:,}")
    print(f"Total Matches:         {total_clean_matches:,} (Avg {total_clean_matches/n_s1:.2f})")
    print(f"Total Singletons:      {clean_singletons:,} ({clean_singletons/n_s1*100:.2f}%)")
    print(f"False Merges Removed:  {sum(len(v) for v in duplicates.values()) - len(duplicates):,}")
    print(f"Total Execution Time:  {time.time()-t0:.2f}s")
    print("="*60)

if __name__ == "__main__":
    main()
