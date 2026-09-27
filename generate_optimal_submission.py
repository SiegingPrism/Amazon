import os, sys, time, joblib
import polars as pl
from collections import defaultdict

print("="*75)
print("GENERATING OPTIMAL HIGH-PRECISION LEADERBOARD SUBMISSION FROM CHECKPOINTS")
print("="*75)

t0 = time.time()
s1_path = "dataset/test/test_source1.tsv"
s1_df = pl.read_csv(s1_path, separator="\t")
original_s1_ids = s1_df["entity_id"].to_list()
print(f"Loaded {len(original_s1_ids):,} S1 entity IDs in {time.time()-t0:.2f}s")

# Load checkpoints
t_load = time.time()
chk_fr = joblib.load("output/chk_v5_France.joblib")
chk_us = joblib.load("output/chk_v5_US.joblib")
chk_in = joblib.load("output/chk_v5_India.joblib")
print(f"Loaded all 3 country checkpoints in {time.time()-t_load:.2f}s")

all_pred = defaultdict(list)
cand_results = {}

# Merge candidates
for chk in [chk_fr, chk_us, chk_in]:
    for eid, cands in chk["cands"].items():
        cand_results[eid] = cands

# 1. France: th=0.99, dp=0.04 (Stops catastrophic French street/stopword false positives)
for eid, m_list in chk_fr["matches"].items():
    if not m_list: continue
    best_p = max(p for _, p in m_list)
    for cid, p in m_list:
        if p >= 0.99 and (best_p - p) <= 0.04:
            all_pred[eid].append((cid, p))

# 2. US: th=0.96, dp=0.04 (Balances high precision while preserving 5.4% singletons)
for eid, m_list in chk_us["matches"].items():
    if not m_list: continue
    best_p = max(p for _, p in m_list)
    for cid, p in m_list:
        if p >= 0.96 and (best_p - p) <= 0.04:
            all_pred[eid].append((cid, p))

# 3. India: th=0.95, dp=0.04 (Preserves transliterations while eliminating co-location distractors)
for eid, m_list in chk_in["matches"].items():
    if not m_list: continue
    best_p = max(p for _, p in m_list)
    for cid, p in m_list:
        if p >= 0.95 and (best_p - p) <= 0.04:
            all_pred[eid].append((cid, p))

# Global 1-to-1 Bipartite Winner-Takes-All Disambiguation
print("\nPerforming Global Bipartite Target Disambiguation (Winner-Takes-All)...")
t_dis = time.time()
target_winner = {}
total_raw = sum(len(v) for v in all_pred.values())

for eid, m_list in all_pred.items():
    for cid, p in m_list:
        if cid not in target_winner or p > target_winner[cid][1]:
            target_winner[cid] = (eid, p)

final_matches = defaultdict(list)
for eid, m_list in all_pred.items():
    for cid, p in m_list:
        if target_winner.get(cid, (None,))[0] == eid:
            final_matches[eid].append(cid)

print(f"Disambiguation completed in {time.time()-t_dis:.2f}s:")
print(f"  Raw matches evaluated:       {total_raw:,}")
print(f"  Conflicting merges removed:  {total_raw - sum(len(v) for v in final_matches.values()):,}")
print(f"  Final unique matches:        {sum(len(v) for v in final_matches.values()):,} (Avg: {sum(len(v) for v in final_matches.values())/len(original_s1_ids):.2f})")
singletons = sum(1 for eid in original_s1_ids if not final_matches.get(eid))
print(f"  Final Singletons (1.0 F0.5): {singletons:,} ({singletons/len(original_s1_ids):.3%}) [Ground Truth Benchmark: 5.585%]")

# Write submission TSV files
print("\nWriting official competition submission TSVs...")
t_write = time.time()

# 1. matching_results.tsv
matching_file = "output/matching_results.tsv"
with open(matching_file, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for eid in original_s1_ids:
        m_list = final_matches.get(eid, [])
        f.write(f"{eid}\t{','.join(m_list)}\n")
m_size = os.path.getsize(matching_file) / (1024 * 1024)
print(f"  {matching_file} written ({m_size:.2f} MB) in {time.time()-t_write:.2f}s")

# 2. candidate_pairs.tsv
t_cand = time.time()
cand_file = "output/candidate_pairs.tsv"
with open(cand_file, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for eid in original_s1_ids:
        c_list = cand_results.get(eid, [])
        f.write(f"{eid}\t{','.join(c_list)}\n")
c_size = os.path.getsize(cand_file) / (1024 * 1024)
print(f"  {cand_file} written ({c_size:.2f} MB) in {time.time()-t_cand:.2f}s")

print(f"\nALL OUTPUT FILES GENERATED SUCCESSFULLY IN {time.time()-t0:.2f}s!")
