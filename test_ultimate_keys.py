import sys, os, re, unicodedata, math
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/code/business_entity_resolution"))

import polars as pl
from collections import defaultdict
from test_new_pipeline import FastInvertedIndex, STATE_MAP, extract_improved_keys
from src.utils import clean_number, LEGAL_STOPWORDS, ADDR_STOPWORDS

# Indic State Names mapped to canonical lowercase English
INDIC_STATE_MAP = {
    'तमिलनाडु': 'tamil nadu', 'தமிழ்நாடு': 'tamil nadu', 'తమిళనాడు': 'tamil nadu', 'ತಮಿಳುನಾಡು': 'tamil nadu',
    'उत्तर प्रदेश': 'uttar pradesh', 'உத்தரப் பிரதேசம்': 'uttar pradesh', 'ఉత్తర ప్రదేశ్': 'uttar pradesh',
    'महाराष्ट्र': 'maharashtra', 'மகாராஷ்டிரா': 'maharashtra', 'మహారాష్ట్ర': 'maharashtra', 'મહારાષ્ટ્ર': 'maharashtra',
    'गुजरात': 'gujarat', 'குஜராத்': 'gujarat', 'గుజరాత్': 'gujarat', 'ગુજરાત': 'gujarat',
    'पश्चिम बंगाल': 'west bengal', 'மேற்கு வங்கம்': 'west bengal', 'পশ্চিমবঙ্গ': 'west bengal',
    'कर्नाटक': 'karnataka', 'கர்நாடகா': 'karnataka', 'కర్ణాటక': 'karnataka', 'ಕರ್ನಾಟಕ': 'karnataka',
    'केरल': 'kerala', 'கேரளா': 'kerala', 'కేరళ': 'kerala', 'കേരളം': 'kerala', 'keralam': 'kerala',
    'राजस्थान': 'rajasthan', 'ராஜஸ்தான்': 'rajasthan', 'రాజస్థాన్': 'rajasthan', 'રાજસ્થાન': 'rajasthan',
    'ओडिशा': 'odisha', 'ஒடிசா': 'odisha', 'ఒడిశా': 'odisha', 'ଓଡ଼ିଶା': 'odisha', 'orissa': 'odisha',
    'तेलंगाना': 'telangana', 'தெலுங்கானா': 'telangana', 'తెలంగాణ': 'telangana',
    'आंध्र प्रदेश': 'andhra pradesh', 'ஆந்திர பிரதேசம்': 'andhra pradesh', 'ఆంధ్ర ప్రదేశ్': 'andhra pradesh',
    'बिहार': 'bihar', 'பீகார்': 'bihar', 'బీహార్': 'bihar',
    'मध्य प्रदेश': 'madhya pradesh', 'மத்திய பிரதேசம்': 'madhya pradesh', 'మధ్య ప్రదేశ్': 'madhya pradesh',
    'पंजाब': 'punjab', 'பஞ்சாப்': 'punjab', 'పంజాబ్': 'punjab', 'ਪੰਜਾਬ': 'punjab',
    'हरियाणा': 'haryana', 'ஹரியானா': 'haryana', 'హర్యానా': 'haryana',
    'दिल्ली': 'delhi', 'டெல்லி': 'delhi', 'ఢిల్లీ': 'delhi'
}

BRAHMI_OFFSET_MAP = {
    0x02: "n", 0x03: "h", 0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ee", 0x09: "u", 0x0A: "oo",
    0x0B: "ri", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x12: "o", 0x13: "o", 0x14: "au",
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "nn",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "rr", 0x32: "l", 0x33: "l", 0x34: "lh", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x3E: "aa", 0x3F: "i", 0x40: "ee", 0x41: "u", 0x42: "oo", 0x43: "ri",
    0x46: "e", 0x47: "e", 0x48: "ai", 0x4A: "o", 0x4B: "o", 0x4C: "au",
    0x4D: "",
}
BLOCK_STARTS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]

def transliterate_indic(text):
    if not text: return ""
    res = []
    for ch in text:
        code = ord(ch)
        matched = False
        for b in BLOCK_STARTS:
            if b <= code < b + 0x80:
                offset = code - b
                if offset in BRAHMI_OFFSET_MAP:
                    res.append(BRAHMI_OFFSET_MAP[offset])
                    matched = True
                    break
        if not matched:
            res.append(ch)
    return "".join(res)

def clean_text_indic(text):
    if not text: return ""
    for k, v in INDIC_STATE_MAP.items():
        if k in text:
            text = text.replace(k, " " + v + " ")
    text = transliterate_indic(text)
    text = unicodedata.normalize('NFKD', text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r'\.(com|org|net|co|in|fr|io|info)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_phonetic_skeleton(word):
    if len(word) < 3 or word in LEGAL_STOPWORDS:
        return ""
    w = word.lower().replace("ph", "f").replace("ee", "i").replace("oo", "u")
    tr = str.maketrans({"g": "k", "d": "t", "b": "p", "v": "w", "z": "s", "j": "s", "c": "k", "q": "k", "x": "ks"})
    w = w.translate(tr)
    # remove vowels including y
    sk = re.sub(r"[aeiouy\s]", "", w)
    sk = re.sub(r"(.)\1+", r"\1", sk)
    return sk if len(sk) >= 2 else ""

def extract_ultimate_keys(name, address):
    clean_n = clean_text_indic(name)
    clean_a = clean_text_indic(address) if address else ''
    has_addr = bool(address)
    keys = []
    
    raw_words = [w for w in clean_n.split() if w not in LEGAL_STOPWORDS]
    words = [w for w in raw_words if len(w) >= 3]
    if not words and raw_words: words = raw_words
        
    if words:
        keys.append('NF:' + '_'.join(words[:3]))
        for w in words[:4]:
            if len(w) >= 3:
                keys.append('NW:' + w)
                if len(w) >= 5:
                    keys.append('NP:' + w[:4])
                sk = get_phonetic_skeleton(w)
                if sk:
                    keys.append('NSK:' + sk)
            elif len(w) == 2:
                keys.append('NW2:' + w)
                
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
        
        # 1. Number + Address word
        if nums and awords:
            sig_words = list(dict.fromkeys(awords[:3] + awords[-3:]))
            for num in nums[:2]:
                for aw in sig_words:
                    keys.append(f'ANW:{num}_{aw}')
                    
        # 2. Address Word pairs
        if len(awords) >= 2:
            keys.append(f'AAW:{awords[0]}_{awords[1]}')
            keys.append(f'AAW:{awords[-2]}_{awords[-1]}')
            if len(awords) >= 3:
                keys.append(f'AAW:{awords[0]}_{awords[-1]}')
                
        # 3. Rare single address words (length >= 6)
        for aw in awords:
            if len(aw) >= 6:
                keys.append(f'RAW:{aw}')
                
        # 4. Name + Address word
        if words and awords:
            keys.append(f'NAW:{words[0]}_{awords[0]}')
            if len(awords) >= 2:
                keys.append(f'NAW:{words[0]}_{awords[-1]}')
                
    return keys

class UltimateInvertedIndex:
    def __init__(self, max_key_len=5000):
        self.index = defaultdict(list)
        self.max_key_len = max_key_len
        self.records = []
        self.idfs = {}

    def add_records(self, eids, names, addrs):
        start_idx = len(self.records)
        for i in range(len(eids)):
            idx = start_idx + i
            self.records.append((eids[i], names[i], addrs[i]))
            keys = set(extract_ultimate_keys(names[i], addrs[i]))
            for k in keys:
                self.index[k].append(idx)
        N = len(self.records)
        for k, posting in self.index.items():
            self.idfs[k] = math.log(1.0 + (N / len(posting)))

    def query(self, name, address, top_name=25, top_addr=15):
        import math
        keys = set(extract_ultimate_keys(name, address))
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

if __name__ == "__main__":
    import math
    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
    s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
    s1_india = s1_full.filter(pl.col("country") == "India")
    india_eids = s1_india["entity_id"][:5000].to_list()

    val_gt = gt_df.filter(pl.col("source1_entity_id").is_in(india_eids))
    val_gt_dict = {r["source1_entity_id"]: set(r["matched_entity_ids"].split(",")) if r["matched_entity_ids"] else set() for r in val_gt.iter_rows(named=True)}
    needed_targets = set.union(*val_gt_dict.values()) if val_gt_dict else set()

    s1_val_dict = {r["entity_id"]: r for r in s1_india.filter(pl.col("entity_id").is_in(india_eids)).iter_rows(named=True)}
    s2_full = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
    s3_full = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")

    s2_pool = pl.concat([s2_full.filter(pl.col("entity_id").is_in(list(needed_targets))), s2_full.filter(pl.col("country")=="India").sample(n=30000, seed=42)]).unique(subset=["entity_id"])
    s3_pool = pl.concat([s3_full.filter(pl.col("entity_id").is_in(list(needed_targets))), s3_full.filter(pl.col("country")=="India").sample(n=30000, seed=42)]).unique(subset=["entity_id"])

    target_records = {}
    for r in s2_pool.iter_rows(named=True): target_records[r["entity_id"]] = r
    for r in s3_pool.iter_rows(named=True): target_records[r["entity_id"]] = r

    idx = UltimateInvertedIndex(max_key_len=5000)
    idx.add_records([r["entity_id"] for r in target_records.values()], [r["business_name"] for r in target_records.values()], [r["business_address"] for r in target_records.values()])

    total_gt = sum(len(v) for v in val_gt_dict.values())
    hits = 0
    total_cands = 0
    for s1_id, gt_ids in val_gt_dict.items():
        s1_r = s1_val_dict[s1_id]
        c_indices = idx.query(s1_r["business_name"], s1_r["business_address"], top_name=25, top_addr=15)
        total_cands += len(c_indices)
        cands = set(idx.records[i][0] for i in c_indices)
        hits += len(gt_ids & cands)

    print(f"Ultimate India Candidate Recall: {hits}/{total_gt} ({hits/total_gt*100:.2f}%)")
    print(f"Average Candidates per Entity:   {total_cands/len(val_gt_dict):.1f}")
