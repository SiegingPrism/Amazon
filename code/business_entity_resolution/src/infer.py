import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import polars as pl
import numpy as np
import time, joblib
from collections import defaultdict
from rapidfuzz import fuzz

from src.utils import clean_text, clean_address
from src.features import compute_features_precleaned
from src.blocking import InvertedIndex

def run_inference(test_dir="dataset/test", output_dir="output", model_path=None):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "matching_model.joblib")
        
    print("="*75)
    print("      COMPETITIVE BUSINESS ENTITY RESOLUTION INFERENCE PIPELINE       ")
    print(f"Test Directory:   {test_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Model Artifact:   {model_path}")
    print("="*75)
    
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()
    
    # Load trained model artifact
    artifact = joblib.load(model_path)
    clf = artifact["model"]
    base_threshold = artifact.get("threshold", 0.80)
    print(f"Model loaded successfully (n_iter={clf.n_iter_}, max_leaves={clf.max_leaf_nodes})")
    
    # Country-calibrated dual thresholds (optimized for macro F0.5 precision weighting)
    # th_addr: threshold when candidate has address
    # th_no_addr: threshold when candidate has missing address (Source 2 missing address recovery)
    COUNTRY_CONFIGS = {
        "France": {
            "th_addr": 0.88,
            "th_no_addr": 0.86,
            "top_name": 22,
            "top_addr": 10
        },
        "US": {
            "th_addr": 0.872,
            "th_no_addr": 0.820,
            "top_name": 22,
            "top_addr": 10
        },
        "India": {
            "th_addr": 0.882,
            "th_no_addr": 0.786,
            "top_name": 22,
            "top_addr": 10
        }
    }
    print(f"Calibrated Configurations:")
    for c, cfg in COUNTRY_CONFIGS.items():
        print(f"  {c}: th_addr={cfg['th_addr']}, th_no_addr={cfg['th_no_addr']}, top_name={cfg['top_name']}, top_addr={cfg['top_addr']}")
    
    # Read test Source 1, Source 2, Source 3
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    
    print(f"Reading test datasets with Polars...")
    t0 = time.time()
    s1_df = pl.read_csv(s1_path, separator="\t")
    s2_df = pl.read_csv(s2_path, separator="\t")
    s3_df = pl.read_csv(s3_path, separator="\t")
    print(f"Loaded: S1 ({len(s1_df):,}), S2 ({len(s2_df):,}), S3 ({len(s3_df):,}) in {time.time()-t0:.2f}s")
    
    original_s1_ids = s1_df["entity_id"].to_list()
    
    cand_results = {}
    match_results = {eid: [] for eid in original_s1_ids}
    
    total_s1_processed = 0
    total_candidates_generated = 0
    
    # -------------------------------------------------------------
    # PROCESS ALL COUNTRIES: FRANCE, US, INDIA
    # -------------------------------------------------------------
    active_countries = ["France", "US", "India"]
    
    for country in active_countries:
        c_t0 = time.time()
        print(f"\n>>> Processing Country: {country} <<<")
        
        s1_c = s1_df.filter(pl.col("country") == country)
        s2_c = s2_df.filter(pl.col("country") == country)
        s3_c = s3_df.filter(pl.col("country") == country)
        n_s1 = len(s1_c)
        c_cfg = COUNTRY_CONFIGS[country]
        top_name = c_cfg["top_name"]
        top_addr = c_cfg["top_addr"]
        th_addr = c_cfg["th_addr"]
        th_no_addr = c_cfg["th_no_addr"]
        print(f"Country {country}: {n_s1:,} S1 entities, {len(s2_c):,} S2 entities, {len(s3_c):,} S3 entities")
        print(f"Parameters: th_addr={th_addr:.2f}, th_no_addr={th_no_addr:.2f}, top_name={top_name}, top_addr={top_addr}")
        
        if n_s1 == 0:
            continue
            
        chk_path = os.path.join(output_dir, f"chk_{country}.joblib")
        if os.path.exists(chk_path):
            print(f"Loading cached checkpoint for {country} from {chk_path}...")
            saved = joblib.load(chk_path)
            for eid, cands in saved["cands"].items():
                cand_results[eid] = cands
            for eid, matches in saved["matches"].items():
                match_results[eid] = matches
            chk_cands = sum(len(v) for v in saved["cands"].values())
            chk_matches = sum(len(v) for v in saved["matches"].values())
            print(f"Loaded {country}: {len(saved['cands']):,} entities, {chk_cands:,} cands, {chk_matches:,} matches in {time.time()-c_t0:.2f}s")
            total_s1_processed += n_s1
            total_candidates_generated += chk_cands
            continue
            
        # Build Dual IDF Inverted Index on S2 + S3 (max_key_len=1500 for optimal precision & speed)
        idx_t0 = time.time()
        idx = InvertedIndex(max_key_len=1500)
        idx.add_records(s2_c["entity_id"].to_list(), s2_c["business_name"].to_list(), s2_c["business_address"].to_list(), country=country)
        idx.add_records(s3_c["entity_id"].to_list(), s3_c["business_name"].to_list(), s3_c["business_address"].to_list(), country=country)
        print(f"Dual IDF Index built in {time.time()-idx_t0:.2f}s with {len(idx.index):,} unique keys and {len(idx.records):,} records")
        
        # Pre-clean S1 entities
        s1_clean_t0 = time.time()
        s1_eids = s1_c["entity_id"].to_list()
        s1_clean_names = [clean_text(n) for n in s1_c["business_name"].to_list()]
        s1_raw_addrs = s1_c["business_address"].to_list()
        s1_clean_addrs = [clean_address(a, country=country) if a else "" for a in s1_raw_addrs]
        s1_has_addrs = [bool(a) for a in s1_raw_addrs]
        print(f"Pre-cleaned {n_s1:,} S1 entities in {time.time()-s1_clean_t0:.2f}s")
        
        c_matches = 0
        c_cands = 0
        
        chunk_size = 25000
        for chunk_start in range(0, n_s1, chunk_size):
            chunk_end = min(n_s1, chunk_start + chunk_size)
            chk_t0 = time.time()
            
            batch_feats = []
            pair_tracking = [] # list of (eid, cid, ha2)
            
            for i in range(chunk_start, chunk_end):
                eid = s1_eids[i]
                cn1 = s1_clean_names[i]
                ca1 = s1_clean_addrs[i]
                ha1 = s1_has_addrs[i]
                
                # Query Dual IDF index with optimal depth
                cand_indices = idx.query_precleaned(cn1, ca1, ha1, top_name=top_name, top_addr=top_addr)
                cand_ids = [idx.records[ci][0] for ci in cand_indices]
                cand_results[eid] = cand_ids
                c_cands += len(cand_ids)
                
                for ci in cand_indices:
                    cid, cn2, ca2, ha2 = idx.records[ci]
                    
                    # Guardrail: skip pair if name similarity is essentially zero
                    cn1_j = cn1.replace(" ", "")
                    cn2_j = cn2.replace(" ", "")
                    joined_ratio = fuzz.ratio(cn1_j, cn2_j)
                    if joined_ratio < 75 and fuzz.token_set_ratio(cn1, cn2) < 25 and fuzz.partial_ratio(cn1, cn2) < 30:
                        continue
                        
                    feats = compute_features_precleaned(cn1, ca1, cn2, ca2, ha2)
                    n_fuzz = feats[0]
                    n_set = feats[2]
                    a_set = feats[7]
                    num_m = feats[10]
                    
                    # Multi-tenant commercial building guard:
                    # Different businesses sharing an office tower/industrial park
                    n_cut = 0.52 if country == "US" else 0.55
                    if n_set < n_cut and n_fuzz < n_cut:
                        continue
                    if num_m == 0.0 and n_set < (0.62 if country == "US" else 0.65):
                        continue
                        
                    batch_feats.append(feats)
                    pair_tracking.append((eid, cid, ha2, feats))
                    
            # Batched prediction
            if batch_feats:
                X = np.array(batch_feats, dtype=np.float32)
                probs = clf.predict_proba(X)[:, 1]
                n_cut = 0.52 if country == "US" else 0.55
                for (eid, cid, ha2, feats), p in zip(pair_tracking, probs):
                    eff_threshold = th_addr if ha2 else th_no_addr
                    
                    # High-confidence name match bonus (recovers true missing-address matches)
                    n_set = feats[2]
                    a_set = feats[7]
                    if n_set >= 0.92 and (a_set >= 0.70 or not ha2):
                        eff_threshold = min(eff_threshold, 0.74)
                        
                    if p >= eff_threshold:
                        match_results[eid].append((cid, float(p)))
                        c_matches += 1
                        
            chk_el = time.time() - chk_t0
            processed_so_far = chunk_end
            print(f"  Processed {processed_so_far:,}/{n_s1:,} ({processed_so_far/n_s1:.1%}) in {chk_el:.2f}s | Speed: {(chunk_end-chunk_start)/chk_el:.0f} entities/s")
            
        country_probs = [p for eid in s1_eids for _, p in match_results[eid]]
        if country_probs:
            print(f"  {country} Probability Profile: mean={np.mean(country_probs):.3f}, median={np.median(country_probs):.3f}, p10={np.percentile(country_probs,10):.3f}, p90={np.percentile(country_probs,90):.3f}")
            
        print(f"Completed {country} raw matching in {time.time()-c_t0:.2f}s (candidates: {c_cands:,}, raw matches: {c_matches:,})")
        
        # Save country checkpoint
        joblib.dump({"cands": {eid: cand_results[eid] for eid in s1_eids}, "matches": {eid: match_results[eid] for eid in s1_eids}}, chk_path)
        print(f"Saved fault-tolerant checkpoint: {chk_path}")
        
        total_s1_processed += n_s1
        total_candidates_generated += c_cands
        del idx
        del s1_c, s2_c, s3_c

    # -------------------------------------------------------------
    # 3. BIPARTITE TARGET DISAMBIGUATION (Winner-Takes-All Argmax)
    # -------------------------------------------------------------
    print("\n" + "="*75)
    print("Performing Competitive Target Disambiguation (Bipartite 1-to-1 Optimization)...")
    print("="*75)
    t_dis0 = time.time()
    
    target_winner = {}
    total_raw_pairs = 0
    for eid, m_list in match_results.items():
        total_raw_pairs += len(m_list)
        for cid, p in m_list:
            if cid not in target_winner or p > target_winner[cid][1]:
                target_winner[cid] = (eid, p)
                
    final_matches = defaultdict(list)
    eliminated_conflicts = 0
    for eid, m_list in match_results.items():
        for cid, p in m_list:
            if target_winner.get(cid, (None,))[0] == eid:
                final_matches[eid].append(cid)
            else:
                eliminated_conflicts += 1
                
    total_matches_predicted = sum(len(v) for v in final_matches.values())
    total_singletons = sum(1 for s1_id in original_s1_ids if len(final_matches[s1_id]) == 0)
    print(f"Disambiguation resolved in {time.time()-t_dis0:.2f}s:")
    print(f"  Raw Matches Evaluated:        {total_raw_pairs:,}")
    print(f"  False Cross-Merges Removed:   {eliminated_conflicts:,}")
    print(f"  Final Unique Matches:         {total_matches_predicted:,} (Avg {total_matches_predicted/total_s1_processed:.2f})")
    print(f"  Final Singletons (1.0 F0.5):  {total_singletons:,} ({total_singletons/total_s1_processed:.2%})")
        
    # -------------------------------------------------------------
    # 4. WRITE SUBMISSION TSV FILES
    # -------------------------------------------------------------
    print("\n" + "="*75)
    print("Writing Official Competition Submission TSV Files...")
    print("="*75)
    
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")
    
    # candidate_pairs.tsv
    t_w0 = time.time()
    with open(cand_path, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in original_s1_ids:
            cands = cand_results.get(s1_id, [])
            f_cand.write(f"{s1_id}\t{','.join(cands)}\n")
            
    cand_size_mb = os.path.getsize(cand_path) / (1024 * 1024)
    print(f"candidate_pairs.tsv written ({cand_size_mb:.2f} MB) in {time.time()-t_w0:.2f}s")
    
    # matching_results.tsv
    t_w1 = time.time()
    with open(match_path, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in original_s1_ids:
            matches = final_matches.get(s1_id, [])
            f_match.write(f"{s1_id}\t{','.join(matches)}\n")
            
    match_size_mb = os.path.getsize(match_path) / (1024 * 1024)
    print(f"matching_results.tsv written ({match_size_mb:.2f} MB) in {time.time()-t_w1:.2f}s")
    
    total_time = time.time() - t_start
    print("\n" + "="*75)
    print(f"INFERENCE COMPLETED in {total_time:.2f}s ({total_time/60:.2f} minutes)!")
    print(f"Total S1 Entities:     {total_s1_processed:,}")
    print(f"Candidate Pairs / S1:  {total_candidates_generated/total_s1_processed:.2f} (Competition Target: 10-50)")
    print(f"Matching Results Size: {match_size_mb:.2f} MB (Portal Limit: < 200 MB)")
    print("="*75)

if __name__ == "__main__":
    run_inference()
