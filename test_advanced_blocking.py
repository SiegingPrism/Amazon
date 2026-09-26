import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/code/business_entity_resolution"))

import polars as pl
import numpy as np
import joblib, re
from collections import defaultdict
from rapidfuzz import fuzz
from src.utils import clean_text, evaluate_entity_f05, clean_number, STATE_MAP, LEGAL_STOPWORDS, ADDR_STOPWORDS

LEET_MAP = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b"}
def deleet_token(tok):
    if re.search(r"[a-zA-Z]", tok) and re.search(r"[0-9]", tok):
        return "".join(LEET_MAP.get(c, c) for c in tok)
    return tok
def deleet_text(text):
    if not text: return ""
    return " ".join(deleet_token(t) for t in text.split())

def get_phonetic_skeleton_v2(word):
    if len(word) < 2: return ""
    w = word.lower().replace("ph", "f").replace("ee", "i").replace("oo", "u")
    tr = str.maketrans({"g": "k", "d": "t", "b": "p", "v": "w", "z": "s", "j": "s", "c": "k", "q": "k", "x": "ks"})
    w = w.translate(tr)
    w = re.sub(r"(?<=[bcdfghjklmnpqrstvwxyz])h", "", w)
    sk = re.sub(r"[aeiouy\s]", "", w)
    sk = re.sub(r"(.)\1+", r"\1", sk)
    return sk if len(sk) >= 2 else ""

def extract_advanced_keys(clean_n, clean_a, has_addr):
    keys = []
    clean_n = deleet_text(clean_n)
    raw_words = [w for w in clean_n.split() if w not in LEGAL_STOPWORDS]
    
    # Keep words length >= 2 so TL, SK, OM, RK, AL are preserved
    words = [w for w in raw_words if len(w) >= 2]
    if not words and raw_words: words = raw_words
    
    if words:
        keys.append("NF:" + "_".join(words[:3]))
        # Concatenated name key (for domains like sunriseinfotech.com, aaratransport.com)
        if len(words) >= 2:
            keys.append("NC:" + "".join(words[:3]))
        elif len(words) == 1 and len(words[0]) >= 6:
            keys.append("NC:" + words[0])
            
        for w in words[:4]:
            if len(w) >= 3:
                keys.append("NW:" + w)
                if len(w) >= 5:
                    keys.append("NP:" + w[:4])
                sk = get_phonetic_skeleton_v2(w)
                if sk:
                    keys.append("NSK:" + sk)
            elif len(w) == 2:
                keys.append("NW2:" + w)
                
    if has_addr and clean_a:
        a_tokens = clean_a.split()
        expanded = []
        for tok in a_tokens:
            if tok in STATE_MAP: expanded.extend(STATE_MAP[tok].split())
            else: expanded.append(tok)
            
        nums = [clean_number(w) for w in expanded if any(c.isdigit() for c in w)]
        nums = [n for n in nums if n]
        awords = [w for w in expanded if w not in ADDR_STOPWORDS and not any(c.isdigit() for c in w) and len(w) >= 3]
        
        if nums and awords:
            sig = list(dict.fromkeys(awords[:3] + awords[-3:]))
            for num in nums[:2]:
                for aw in sig:
                    keys.append(f"ANW:{num}_{aw}")
                    
        if len(awords) >= 2:
            keys.append(f"AAW:{awords[0]}_{awords[1]}")
            keys.append(f"AAW:{awords[-2]}_{awords[-1]}")
            if len(awords) >= 3:
                keys.append(f"AAW:{awords[0]}_{awords[-1]}")
                
        for aw in awords:
            if len(aw) >= 6:
                keys.append(f"RAW:{aw}")
                
        if words and awords:
            keys.append(f"NAW:{words[0]}_{awords[0]}")
            if len(awords) >= 2:
                keys.append(f"NAW:{words[0]}_{awords[-1]}")
                
    return keys

class AdvancedInvertedIndex:
    def __init__(self, max_key_len=2000):
        self.index = defaultdict(list)
        self.max_key_len = max_key_len
        self.records = []
        self.idfs = {}

    def add_records(self, eids, names, addrs):
        start_idx = len(self.records)
        clean_names = [clean_text(deleet_text(n)) for n in names]
        clean_addrs = [clean_text(a) if a else "" for a in addrs]
        has_addrs = [bool(a) for a in addrs]
        for i in range(len(eids)):
            idx = start_idx + i
            cn, ca, ha = clean_names[i], clean_addrs[i], has_addrs[i]
            self.records.append((eids[i], cn, ca, ha))
            keys = set(extract_advanced_keys(cn, ca, ha))
            for k in keys:
                self.index[k].append(idx)
        N = len(self.records)
        for k, plist in self.index.items():
            if len(plist) <= self.max_key_len:
                self.idfs[k] = math.log(1.0 + N / len(plist))

    def query(self, name, addr, top_name=25, top_addr=15):
        cn = clean_text(deleet_text(name))
        ca = clean_text(addr) if addr else ""
        keys = set(extract_advanced_keys(cn, ca, bool(addr)))
        name_scores = defaultdict(float)
        addr_scores = defaultdict(float)
        for k in keys:
            if k in self.idfs:
                w = self.idfs[k]
                is_name = k.startswith(("NF:", "NW:", "NP:", "NW2:", "NSK:", "NC:"))
                target_dict = name_scores if is_name else addr_scores
                for idx in self.index[k]:
                    target_dict[idx] += w
        n_cands = sorted(name_scores, key=name_scores.get, reverse=True)[:top_name]
        a_cands = sorted(addr_scores, key=addr_scores.get, reverse=True)[:top_addr]
        return list(dict.fromkeys(n_cands + a_cands))

import math

print("="*75)
print("BENCHMARKING ADVANCED INDEX & FEATURES ON 10,000 VALIDATION ENTITIES")
print("="*75)

gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t").slice(10000, 10000)
s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
s2_full = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
s3_full = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")

gt_dict = {r["source1_entity_id"]: set(r["matched_entity_ids"].split(",")) if r["matched_entity_ids"] else set() for r in gt_df.iter_rows(named=True)}
needed = set.union(*gt_dict.values())

s1_dict = {r["entity_id"]: r for r in s1_full.filter(pl.col("entity_id").is_in(list(gt_dict.keys()))).iter_rows(named=True)}
s2_pool = pl.concat([s2_full.filter(pl.col("entity_id").is_in(list(needed))), s2_full.sample(n=100000, seed=42)]).unique(subset=["entity_id"])
s3_pool = pl.concat([s3_full.filter(pl.col("entity_id").is_in(list(needed))), s3_full.sample(n=100000, seed=42)]).unique(subset=["entity_id"])

target_records = {}
for r in s2_pool.iter_rows(named=True): target_records[r["entity_id"]] = r
for r in s3_pool.iter_rows(named=True): target_records[r["entity_id"]] = r

indices = {"US": AdvancedInvertedIndex(max_key_len=2000), "India": AdvancedInvertedIndex(max_key_len=2000)}
recs_by_c = defaultdict(lambda: ([], [], []))
for tid, tr in target_records.items():
    c = tr["country"]
    if c in indices:
        recs_by_c[c][0].append(tr["entity_id"])
        recs_by_c[c][1].append(tr["business_name"])
        recs_by_c[c][2].append(tr["business_address"])

for c in indices:
    eids, names, addrs = recs_by_c[c]
    indices[c].add_records(eids, names, addrs)

total_gt = sum(len(v) for v in gt_dict.values())
recalled = 0
total_cands = 0

for s1_id, gt in gt_dict.items():
    s1_r = s1_dict[s1_id]
    c = s1_r["country"]
    if c in indices:
        c_indices = indices[c].query(s1_r["business_name"], s1_r["business_address"], top_name=25, top_addr=15)
        cand_eids = set(indices[c].records[i][0] for i in c_indices)
        recalled += len(gt & cand_eids)
        total_cands += len(cand_eids)

print(f"NEW ADVANCED BLOCKING RECALL: {recalled}/{total_gt} ({recalled/total_gt*100:.3f}%) | Avg cands: {total_cands/len(gt_dict):.2f}")
