import re, math
from collections import defaultdict
from .utils import clean_text, clean_address, clean_number, LEGAL_STOPWORDS, ADDR_STOPWORDS, STATE_MAP, get_phonetic_skeleton

def extract_blocking_keys(name, address, country=None):
    clean_n = clean_text(name)
    clean_a = clean_address(address, country=country) if address else ""
    return extract_blocking_keys_precleaned(clean_n, clean_a, bool(address))

def extract_blocking_keys_precleaned(clean_n, clean_a, has_addr):
    keys = []
    raw_words = [w for w in clean_n.split() if w not in LEGAL_STOPWORDS]
    words = [w for w in raw_words if len(w) >= 2]
    if not words and raw_words:
        words = raw_words
        
    # 1. Name keys
    if words:
        keys.append("NF:" + "_".join(words[:3]))
        if len(words) >= 2:
            keys.append("NC:" + "".join(words[:3]))
        elif len(words) == 1 and len(words[0]) >= 6:
            keys.append("NC:" + words[0])
            
        for w in words[:4]:
            if len(w) >= 3:
                keys.append("NW:" + w)
                if len(w) >= 5:
                    keys.append("NP:" + w[:4])
                sk = get_phonetic_skeleton(w)
                if sk:
                    keys.append("NSK:" + sk)
            elif len(w) == 2:
                keys.append("NW2:" + w)
                
    # 2. Address keys
    if has_addr and clean_a:
        a_tokens = clean_a.split()
        expanded_tokens = []
        for tok in a_tokens:
            if tok in STATE_MAP:
                expanded_tokens.extend(STATE_MAP[tok].split())
            else:
                expanded_tokens.append(tok)
                
        nums = [clean_number(w) for w in expanded_tokens if any(c.isdigit() for c in w)]
        nums = [n for n in nums if n]
        awords = [w for w in expanded_tokens if w not in ADDR_STOPWORDS and not any(c.isdigit() for c in w) and len(w) >= 3]
        
        # Compound key: number + significant address words
        if nums and awords:
            sig_words = list(dict.fromkeys(awords[:3] + awords[-3:]))
            for num in nums[:2]:
                for aw in sig_words:
                    keys.append(f"ANW:{num}_{aw}")
                    
        # Word pair inside address (e.g. street + city)
        if len(awords) >= 2:
            keys.append(f"AAW:{awords[0]}_{awords[1]}")
            keys.append(f"AAW:{awords[-2]}_{awords[-1]}")
            if len(awords) >= 3:
                keys.append(f"AAW:{awords[0]}_{awords[-1]}")
                
        # Rare single address words (length >= 6)
        for aw in awords:
            if len(aw) >= 6:
                keys.append(f"RAW:{aw}")
                
        # First name word + address word
        if words and awords:
            keys.append(f"NAW:{words[0]}_{awords[0]}")
            if len(awords) >= 2:
                keys.append(f"NAW:{words[0]}_{awords[-1]}")
                
    return keys

class InvertedIndex:
    """
    High-Recall Dual IDF Inverted Index for candidate generation.
    Ranks candidates by Inverse Document Frequency (IDF) of shared blocking keys.
    """
    def __init__(self, max_key_len=5000):
        self.index = defaultdict(list)
        self.max_key_len = max_key_len
        self.records = []
        self.idfs = {}

    def add_records(self, eids, names, addrs, country=None):
        start_idx = len(self.records)
        clean_names = [clean_text(n) for n in names]
        clean_addrs = [clean_address(a, country=country) if a else "" for a in addrs]
        has_addrs = [bool(a) for a in addrs]
        
        for i in range(len(eids)):
            idx = start_idx + i
            cn = clean_names[i]
            ca = clean_addrs[i]
            ha = has_addrs[i]
            self.records.append((eids[i], cn, ca, ha))
            keys = set(extract_blocking_keys_precleaned(cn, ca, ha))
            for k in keys:
                self.index[k].append(idx)
                
        # Precompute key IDFs
        N = len(self.records)
        for k, posting in self.index.items():
            self.idfs[k] = math.log(1.0 + (N / len(posting)))

    def query_precleaned(self, clean_n, clean_a, has_addr, top_name=25, top_addr=15):
        keys = set(extract_blocking_keys_precleaned(clean_n, clean_a, has_addr))
        name_scores = {}
        addr_scores = {}
        
        for k in keys:
            posting = self.index.get(k)
            if posting and len(posting) <= self.max_key_len:
                idf = self.idfs.get(k, 1.0)
                if k.startswith(('NF:', 'NW:', 'NW2:', 'NP:', 'NSK:', 'NAW:')):
                    mult = 2.0 if k.startswith(('NF:', 'NSK:')) else 1.0
                    for i in posting:
                        name_scores[i] = name_scores.get(i, 0.0) + (idf * mult)
                if k.startswith(('ANW:', 'AAW:', 'RAW:', 'NAW:')):
                    mult = 1.5 if k.startswith(('ANW:', 'RAW:')) else 1.0
                    for i in posting:
                        addr_scores[i] = addr_scores.get(i, 0.0) + (idf * mult)
                        
        top_n = sorted(name_scores, key=name_scores.get, reverse=True)[:top_name]
        top_a = sorted(addr_scores, key=addr_scores.get, reverse=True)[:top_addr]
        return list(dict.fromkeys(top_n + top_a))

    def query(self, name, address, top_k=20):
        clean_n = clean_text(name)
        clean_a = clean_text(address) if address else ""
        return self.query_precleaned(clean_n, clean_a, bool(address))
