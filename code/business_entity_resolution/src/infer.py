import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import polars as pl
import numpy as np
import time, joblib
from collections import defaultdict, Counter
from rapidfuzz import fuzz

from src.utils import clean_text, clean_address, LEGAL_STOPWORDS
from src.features_v2 import compute_features_v2_precleaned
from src.blocking import InvertedIndex

def run_inference(test_dir="dataset/test", output_dir="output"):
    print("="*75)
    print(" COMPETITIVE BUSINESS ENTITY RESOLUTION PIPELINE (V4 - HIGH PRECISION) ")
    print(f"Test Directory:   {test_dir}")
    print(f"Output Directory: {output_dir}")
    print("="*75)
    
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()
    
    # Load dedicated country models
    src_dir = os.path.dirname(__file__)
    us_model_path = os.path.join(src_dir, "model_US.joblib")
    india_model_path = os.path.join(src_dir, "model_India.joblib")
    
    print(f"Loading dedicated US model from: {us_model_path}")
    clf_us = joblib.load(us_model_path)["model"]
    print(f"Loading dedicated India model from: {india_model_path}")
    clf_india = joblib.load(india_model_path)["model"]
    
    models = {
        "US": clf_us,
        "France": clf_us, # High-precision transfer for Latin-script European entities
        "India": clf_india
    }
    
    # Optimized Country Configurations (Dual-Track V5 Engine)
    COUNTRY_CONFIGS = {
        "France": {
            "threshold": 0.94,
            "delta_p": 0.08,
            "core_min": 0.40,
            "max_k": 10,
            "top_name": 25,
            "top_addr": 12
        },
        "US": {
            "threshold": 0.88,
            "delta_p": 0.08,
            "core_min": 0.35,
            "max_k": 10,
            "top_name": 25,
            "top_addr": 12
        },
        "India": {
            "threshold": 0.87,
            "delta_p": 0.08,
            "core_min": 0.30,
            "max_k": 10,
            "top_name": 25,
            "top_addr": 12
        }
    }
    print(f"Calibrated Configurations:")
    for c, cfg in COUNTRY_CONFIGS.items():
        print(f"  {c}: th={cfg['threshold']}, delta_p={cfg['delta_p']}, core_min={cfg['core_min']}, max_k={cfg['max_k']}")
    
    # Read test Source 1, Source 2, Source 3
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    
    print(f"\nReading test datasets with Polars...")
    t0 = time.time()
    s1_df = pl.read_csv(s1_path, separator="\t")
    s2_df = pl.read_csv(s2_path, separator="\t")
    s3_df = pl.read_csv(s3_path, separator="\t")
    print(f"Loaded: S1 ({len(s1_df):,}), S2 ({len(s2_df):,}), S3 ({len(s3_df):,}) in {time.time()-t0:.2f}s")
    
    original_s1_ids = s1_df["entity_id"].to_list()
    cand_results = {}
    match_results = defaultdict(list)
    
    total_s1_processed = 0
    total_candidates_generated = 0
    
    # Process Countries
    active_countries = ["France", "US", "India"]
    
    for country in active_countries:
        c_t0 = time.time()
        print(f"\n{'='*75}")
        print(f">>> Processing Country: {country} <<<")
        print(f"{'='*75}")
        
        s1_c = s1_df.filter(pl.col("country") == country)
        s2_c = s2_df.filter(pl.col("country") == country)
        s3_c = s3_df.filter(pl.col("country") == country)
        n_s1 = len(s1_c)
        c_cfg = COUNTRY_CONFIGS[country]
        clf = models[country]
        
        top_name = c_cfg["top_name"]
        top_addr = c_cfg["top_addr"]
        th = c_cfg["threshold"]
        delta_p = c_cfg["delta_p"]
        core_min = c_cfg["core_min"]
        max_k = c_cfg["max_k"]
        
        print(f"Country {country}: {n_s1:,} S1 entities, {len(s2_c):,} S2 entities, {len(s3_c):,} S3 entities")
        
        if n_s1 == 0:
            continue
            
        chk_path = os.path.join(output_dir, f"chk_v5_{country}.joblib")
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
            
        # Build Inverted Index on S2 + S3
        idx_t0 = time.time()
        idx = InvertedIndex(max_key_len=1500)
        idx.add_records(s2_c["entity_id"].to_list(), s2_c["business_name"].to_list(), s2_c["business_address"].to_list(), country=country)
        idx.add_records(s3_c["entity_id"].to_list(), s3_c["business_name"].to_list(), s3_c["business_address"].to_list(), country=country)
        print(f"Inverted Index built in {time.time()-idx_t0:.2f}s with {len(idx.index):,} unique keys and {len(idx.records):,} records")
        
        # Pre-clean S1 entities
        s1_clean_t0 = time.time()
        s1_eids = s1_c["entity_id"].to_list()
        s1_clean_names = [clean_text(n) for n in s1_c["business_name"].to_list()]
        s1_raw_addrs = s1_c["business_address"].to_list()
        s1_clean_addrs = [clean_address(a, country=country) if a else "" for a in s1_raw_addrs]
        s1_has_addrs = [bool(a) for a in s1_raw_addrs]
        addr_freq = Counter(a for a in s1_clean_addrs if a)
        print(f"Pre-cleaned {n_s1:,} S1 entities in {time.time()-s1_clean_t0:.2f}s (unique addresses: {len(addr_freq):,})")
        
        c_matches = 0
        c_cands = 0
        
        chunk_size = 25000
        for chunk_start in range(0, n_s1, chunk_size):
            chunk_end = min(n_s1, chunk_start + chunk_size)
            chk_t0 = time.time()
            
            batch_feats = []
            pair_tracking = []
            chunk_scored = defaultdict(list)
            
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
                    
                    # Fast preliminary filter
                    cn1_j = cn1.replace(" ", "")
                    cn2_j = cn2.replace(" ", "")
                    joined_ratio = fuzz.ratio(cn1_j, cn2_j)
                    
                    # Allow if basic name match OR potential address match
                    if joined_ratio < 70 and fuzz.token_set_ratio(cn1, cn2) < 25 and fuzz.partial_ratio(cn1, cn2) < 30:
                        if not (ha1 and ha2 and ca1 and ca2 and fuzz.token_set_ratio(ca1[:30], ca2[:30]) >= 70):
                            continue
                            
                    feats = compute_features_v2_precleaned(cn1, ca1, cn2, ca2, ha2)
                    c_token_set = feats[8] # core brand token set ratio
                    sk_sim = feats[16]      # phonetic skeleton similarity
                    addr_sim = feats[12]    # address token set ratio
                    num_match = feats[15]   # street number match
                    
                    # Track 1: Brand-Led Matching
                    is_brand = (c_token_set >= core_min or sk_sim >= 0.80 or joined_ratio >= 70)
                    
                    # Track 2: Unique Location Matching (DBAs / Aliases like Gildcalo, Wexveo, Zephdrex, URLs)
                    is_unique_loc = (
                        ha1 and ha2
                        and num_match == 1.0
                        and addr_sim >= 0.88
                        and feats[14] >= 0.40
                        and (addr_freq.get(ca1, 1) <= 2)
                    )
                    
                    if not (is_brand or is_unique_loc):
                        continue
                        
                    batch_feats.append(feats)
                    pair_tracking.append((eid, cid, is_unique_loc))
                    
            # Batched prediction with dedicated country model
            if batch_feats:
                X = np.array(batch_feats, dtype=np.float32)
                probs = clf.predict_proba(X)[:, 1]
                for (eid, cid, is_ul), p in zip(pair_tracking, probs):
                    chunk_scored[eid].append((cid, float(p), is_ul))
                    
                # Apply Relative Drop Filter & Cluster Size Cap per entity
                for eid, clist in chunk_scored.items():
                    if not clist:
                        continue
                    best_prob = max(p for _, p, _ in clist)
                    for cid, p, is_ul in clist:
                        eff_th = min(th, 0.82) if is_ul else th
                        if p >= eff_th and (best_prob - p) <= delta_p:
                            match_results[eid].append((cid, p))
                            c_matches += 1
                    if len(match_results[eid]) > max_k:
                        match_results[eid].sort(key=lambda x: -x[1])
                        match_results[eid] = match_results[eid][:max_k]
                        
            chk_el = time.time() - chk_t0
            processed_so_far = chunk_end
            print(f"  Processed {processed_so_far:,}/{n_s1:,} ({processed_so_far/n_s1:.1%}) in {chk_el:.2f}s | Speed: {(chunk_end-chunk_start)/chk_el:.0f} entities/s")
            
        print(f"Completed {country} raw matching in {time.time()-c_t0:.2f}s (candidates: {c_cands:,}, raw matches: {c_matches:,})")
        
        # Save fault-tolerant country checkpoint
        joblib.dump({
            "cands": {eid: cand_results[eid] for eid in s1_eids},
            "matches": {eid: match_results[eid] for eid in s1_eids}
        }, chk_path)
        print(f"Saved checkpoint: {chk_path}")
        
        total_s1_processed += n_s1
        total_candidates_generated += c_cands
        del idx
        del s1_c, s2_c, s3_c
        
    # Global Bipartite Target Disambiguation (Winner-Takes-All Argmax)
    print("\n" + "="*75)
    print("Performing Global Bipartite Target Disambiguation (1-to-1 Mapping)...")
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
    print(f"  Conflicting Merges Removed:   {eliminated_conflicts:,}")
    print(f"  Final Unique Matches:         {total_matches_predicted:,} (Avg {total_matches_predicted/total_s1_processed:.2f})")
    print(f"  Final Singletons (1.0 F0.5):  {total_singletons:,} ({total_singletons/total_s1_processed:.3%}) [Ground Truth: 5.585%]")
    
    # Write Official Competition TSV Files
    print("\n" + "="*75)
    print("Writing Official Competition Submission TSV Files...")
    print("="*75)
    
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")
    
    # Write candidate_pairs.tsv
    t_w0 = time.time()
    with open(cand_path, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in original_s1_ids:
            cands = cand_results.get(s1_id, [])
            f_cand.write(f"{s1_id}\t{','.join(cands)}\n")
    cand_size_mb = os.path.getsize(cand_path) / (1024 * 1024)
    print(f"candidate_pairs.tsv written ({cand_size_mb:.2f} MB) in {time.time()-t_w0:.2f}s")
    
    # Write matching_results.tsv
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
    print(f"FULL PRODUCTION INFERENCE COMPLETED IN {total_time:.2f}s ({total_time/60:.2f} minutes)!")
    print(f"Total S1 Entities:     {total_s1_processed:,}")
    print(f"Candidate Pairs / S1:  {total_candidates_generated/total_s1_processed:.2f}")
    print(f"Matching Results Size: {match_size_mb:.2f} MB (Portal Limit: < 200 MB)")
    print("="*75)

if __name__ == "__main__":
    run_inference()
