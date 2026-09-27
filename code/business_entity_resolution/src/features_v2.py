from rapidfuzz import fuzz
from .utils import clean_text, clean_number, get_phonetic_skeleton, LEGAL_STOPWORDS, ADDR_STOPWORDS

def compute_features_v2_precleaned(n1_clean, a1_clean, n2_clean, a2_clean, has_addr2):
    """
    Computes a 21-dimensional feature vector between an S1 entity and an S2/S3 candidate.
    Incorporates full-name lexical metrics, core-brand isolation (legal stopword removal),
    co-location penalty features, and phonetic skeleton representations.
    """
    # 1. Full-name lexical metrics
    n1_j = n1_clean.replace(" ", "")
    n2_j = n2_clean.replace(" ", "")
    n_joined_ratio = fuzz.ratio(n1_j, n2_j) / 100.0
    
    n_fuzz_ratio = max(fuzz.ratio(n1_clean, n2_clean) / 100.0, n_joined_ratio)
    n_token_sort = max(fuzz.token_sort_ratio(n1_clean, n2_clean) / 100.0, n_joined_ratio)
    n_token_set = max(fuzz.token_set_ratio(n1_clean, n2_clean) / 100.0, n_joined_ratio)
    n_partial = fuzz.partial_ratio(n1_clean, n2_clean) / 100.0
    
    w1 = [w for w in n1_clean.split() if w not in LEGAL_STOPWORDS and len(w) >= 2]
    w2 = [w for w in n2_clean.split() if w not in LEGAL_STOPWORDS and len(w) >= 2]
    s_w1, s_w2 = set(w1), set(w2)
    
    if n1_j == n2_j or n_joined_ratio >= 0.95:
        n_jaccard = 1.0
        n_overlap = 1.0
    else:
        n_jaccard = len(s_w1 & s_w2) / max(1, len(s_w1 | s_w2)) if (s_w1 or s_w2) else 0.0
        min_len = min(len(s_w1), len(s_w2))
        n_overlap = (len(s_w1 & s_w2) / min_len) if min_len > 0 else 0.0
    
    # 2. Core brand metrics (without legal entity stopwords)
    core1 = " ".join(w1) if w1 else n1_clean
    core2 = " ".join(w2) if w2 else n2_clean
    core1_j = core1.replace(" ", "")
    core2_j = core2.replace(" ", "")
    c_joined_ratio = fuzz.ratio(core1_j, core2_j) / 100.0
    
    c_fuzz_ratio = max(fuzz.ratio(core1, core2) / 100.0, c_joined_ratio)
    c_token_sort = max(fuzz.token_sort_ratio(core1, core2) / 100.0, c_joined_ratio)
    c_token_set = max(fuzz.token_set_ratio(core1, core2) / 100.0, c_joined_ratio)
    c_jaccard = n_jaccard
    c_overlap = n_overlap
    
    # 3. Address features
    addr_null = 0.0 if has_addr2 else 1.0
    if not has_addr2 or not a2_clean:
        a_token_set = 0.0
        a_token_sort = 0.0
        a_jaccard = 0.0
        num_match = 0.5
    else:
        a_token_set = fuzz.token_set_ratio(a1_clean, a2_clean) / 100.0
        a_token_sort = fuzz.token_sort_ratio(a1_clean, a2_clean) / 100.0
        
        aw1 = set(w for w in a1_clean.split() if w not in ADDR_STOPWORDS and len(w) >= 3)
        aw2 = set(w for w in a2_clean.split() if w not in ADDR_STOPWORDS and len(w) >= 3)
        a_jaccard = len(aw1 & aw2) / max(1, len(aw1 | aw2)) if (aw1 or aw2) else 0.0
        
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
            
    # 4. Phonetic skeleton similarity
    sk1 = "".join([get_phonetic_skeleton(w) for w in w1[:3]])
    sk2 = "".join([get_phonetic_skeleton(w) for w in w2[:3]])
    sk_sim = fuzz.ratio(sk1, sk2) / 100.0 if (sk1 and sk2) else 0.0
    
    # 5. Non-linear interaction & co-location penalty features
    name_x_addr = c_token_set * (a_token_set if not addr_null else c_token_set)
    max_sim = max(c_token_set, a_token_set)
    min_sim = min(c_token_set, a_token_set)
    # Co-location penalty: High when address matches strongly but brand names are distinct
    co_location_penalty = a_token_set * (1.0 - c_token_set)
    
    return [
        n_fuzz_ratio, n_token_sort, n_token_set, n_partial, n_jaccard, n_overlap,
        c_fuzz_ratio, c_token_sort, c_token_set, c_jaccard, c_overlap,
        addr_null, a_token_set, a_token_sort, a_jaccard, num_match,
        sk_sim, name_x_addr, max_sim, min_sim, co_location_penalty
    ]
