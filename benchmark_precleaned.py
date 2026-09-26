import sys, os
sys.path.insert(0, os.path.abspath('code/business_entity_resolution'))

import polars as pl
import time, joblib
import numpy as np
from rapidfuzz import fuzz
from src.utils import clean_text, clean_number, LEGAL_STOPWORDS, ADDR_STOPWORDS
from src.blocking import extract_blocking_keys

def compute_features_fast(n1_clean, a1_clean, n2_clean, a2_clean, addr2_raw):
    n_fuzz_ratio = fuzz.ratio(n1_clean, n2_clean) / 100.0
    n_token_sort = fuzz.token_sort_ratio(n1_clean, n2_clean) / 100.0
    n_token_set = fuzz.token_set_ratio(n1_clean, n2_clean) / 100.0
    n_partial = fuzz.partial_ratio(n1_clean, n2_clean) / 100.0
    
    w1 = set(w for w in n1_clean.split() if w not in LEGAL_STOPWORDS and len(w) >= 3)
    w2 = set(w for w in n2_clean.split() if w not in LEGAL_STOPWORDS and len(w) >= 3)
    n_jaccard = len(w1 & w2) / max(1, len(w1 | w2))
    n_overlap = float(len(w1 & w2))
    
    addr_null = 1.0 if not addr2_raw else 0.0
    if not addr2_raw:
        a_token_set = 0.0
        a_token_sort = 0.0
        a_jaccard = 0.0
        num_match = 0.5
    else:
        a_token_set = fuzz.token_set_ratio(a1_clean, a2_clean) / 100.0
        a_token_sort = fuzz.token_sort_ratio(a1_clean, a2_clean) / 100.0
        
        aw1 = set(w for w in a1_clean.split() if w not in ADDR_STOPWORDS and len(w) >= 3)
        aw2 = set(w for w in a2_clean.split() if w not in ADDR_STOPWORDS and len(w) >= 3)
        a_jaccard = len(aw1 & aw2) / max(1, len(aw1 | aw2))
        
        nums1 = set(clean_number(w) for w in a1_clean.split() if any(c.isdigit() for c in w))
        nums1.discard("")
        nums2 = set(clean_number(w) for w in a2_clean.split() if any(c.isdigit() for c in w))
        nums2.discard("")
        
        if not nums1 or not nums2:
            num_match = 0.5
        elif nums1 & nums2:
            num_match = 1.0
        else:
            num_match = 0.0
            
    name_x_addr = n_token_set * (a_token_set if not addr_null else n_token_set)
    max_sim = max(n_token_set, a_token_set)
    min_sim = min(n_token_set, a_token_set)
    
    return [
        n_fuzz_ratio, n_token_sort, n_token_set, n_partial, n_jaccard, n_overlap,
        addr_null, a_token_set, a_token_sort, a_jaccard, num_match,
        name_x_addr, max_sim, min_sim
    ]

# Benchmark precleaned feature extraction
s1_fr = pl.read_csv("dataset/test/test_source1.tsv", separator="\t").filter(pl.col("country") == "France").head(5000)
s2_fr = pl.read_csv("dataset/test/test_source2.tsv", separator="\t").filter(pl.col("country") == "France").head(50000)

s1_names = [clean_text(x) for x in s1_fr["business_name"].to_list()]
s1_addrs = [clean_text(x) for x in s1_fr["business_address"].to_list()]

s2_names = [clean_text(x) for x in s2_fr["business_name"].to_list()]
s2_addrs = [clean_text(x) for x in s2_fr["business_address"].to_list()]
s2_raw_addrs = s2_fr["business_address"].to_list()

t0 = time.time()
N = 50000
for i in range(N):
    compute_features_fast(s1_names[i % 5000], s1_addrs[i % 5000], s2_names[i], s2_addrs[i], s2_raw_addrs[i])
el = time.time() - t0
print(f"Computed {N} precleaned feature vectors in {el:.2f}s -> {N/el:.0f} pairs/sec!")
