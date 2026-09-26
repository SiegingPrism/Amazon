from rapidfuzz import fuzz
from .utils import clean_text, clean_number, LEGAL_STOPWORDS, ADDR_STOPWORDS

def compute_features(s1_name, s1_addr, s2_name, s2_addr):
    """
    Computes a 14-dimensional feature vector between an S1 entity and an S2/S3 candidate.
    """
    n1_clean = clean_text(s1_name)
    n2_clean = clean_text(s2_name)
    a1_clean = clean_text(s1_addr) if s1_addr else ""
    a2_clean = clean_text(s2_addr) if s2_addr else ""
    has_addr2 = bool(s2_addr)
    return compute_features_precleaned(n1_clean, a1_clean, n2_clean, a2_clean, has_addr2)

def compute_features_precleaned(n1_clean, a1_clean, n2_clean, a2_clean, has_addr2):
    """
    Fast feature computation using pre-cleaned strings.
    """
    n1_j = n1_clean.replace(" ", "")
    n2_j = n2_clean.replace(" ", "")
    n_joined_ratio = fuzz.ratio(n1_j, n2_j) / 100.0
    
    n_fuzz_ratio = max(fuzz.ratio(n1_clean, n2_clean) / 100.0, n_joined_ratio)
    n_token_sort = max(fuzz.token_sort_ratio(n1_clean, n2_clean) / 100.0, n_joined_ratio)
    n_token_set = max(fuzz.token_set_ratio(n1_clean, n2_clean) / 100.0, n_joined_ratio)
    n_partial = fuzz.partial_ratio(n1_clean, n2_clean) / 100.0
    
    w1 = set(w for w in n1_clean.split() if w not in LEGAL_STOPWORDS and len(w) >= 2)
    w2 = set(w for w in n2_clean.split() if w not in LEGAL_STOPWORDS and len(w) >= 2)
    
    if n1_j == n2_j or n_joined_ratio >= 0.95:
        n_jaccard = 1.0
        n_overlap = 1.0
    else:
        n_jaccard = len(w1 & w2) / max(1, len(w1 | w2)) if (w1 or w2) else 0.0
        min_len = min(len(w1), len(w2))
        n_overlap = (len(w1 & w2) / min_len) if min_len > 0 else 0.0
    
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
