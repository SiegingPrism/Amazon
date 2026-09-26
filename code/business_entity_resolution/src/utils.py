import re
import unicodedata
import numpy as np

# Phonetic Brahmi mapping for Indic scripts
BRAHMI_OFFSET_MAP = {
    0x01: "n",   # candrabindu (ँ)
    0x02: "n", 0x03: "h", 0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ee", 0x09: "u", 0x0A: "oo",
    0x0B: "ri", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", # 0x11: candra o (ऑ)
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "nn",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "rr", 0x32: "l", 0x33: "l", 0x34: "lh", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x3C: "",    # nukta (़, ਼)
    0x3E: "aa", 0x3F: "i", 0x40: "ee", 0x41: "u", 0x42: "oo", 0x43: "ri",
    0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", # 0x49: candra o sign (ॉ)
    0x4D: "",    # virama (halant)
    0x4F: "o",   # short o / Kashmiri o
    0x50: "om",  # om (ॐ)
    0x56: "ai",  # Dravidian ai
    0x57: "au",  # Dravidian au length mark (ൗ)
    0x5F: "y",   # Odia ya (ୟ)
    0x70: "n",   # Gurmukhi tippi (ੰ)
    0x71: "w",   # Odia wa (ୱ)
    0x74: "onkar", # Gurmukhi Ek Onkar (ੴ)
    0x7A: "n",   # Malayalam chillu nn (ൺ)
    0x7B: "n",   # Malayalam chillu n (ൻ)
    0x7C: "r",   # Malayalam chillu r (ർ)
    0x7D: "l",   # Malayalam chillu l (ൽ)
    0x7E: "l",   # Malayalam chillu ll (ൾ)
    0x7F: "k",   # Malayalam chillu k (ൿ)
}
BLOCK_STARTS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]

def transliterate_indic(text):
    if not text:
        return ""
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

# State abbreviations expansion map
STATE_MAP = {
    # US
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas', 'ca': 'california',
    'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware', 'fl': 'florida', 'ga': 'georgia',
    'hi': 'hawaii', 'id': 'idaho', 'il': 'illinois', 'in': 'indiana', 'ia': 'iowa',
    'ks': 'kansas', 'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi', 'mo': 'missouri',
    'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada', 'nh': 'new hampshire', 'nj': 'new jersey',
    'nm': 'new mexico', 'ny': 'new york', 'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio',
    'ok': 'oklahoma', 'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
    'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah', 'vt': 'vermont',
    'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia', 'wi': 'wisconsin', 'wy': 'wyoming',
    # India
    'ap': 'andhra pradesh', 'ar': 'arunachal pradesh', 'as': 'assam', 'br': 'bihar',
    'cg': 'chhattisgarh', 'ct': 'chhattisgarh', 'ga': 'goa', 'gj': 'gujarat',
    'hr': 'haryana', 'hp': 'himachal pradesh', 'jh': 'jharkhand', 'ka': 'karnataka',
    'kl': 'kerala', 'mp': 'madhya pradesh', 'mh': 'maharashtra', 'mn': 'manipur',
    'ml': 'meghalaya', 'mz': 'mizoram', 'nl': 'nagaland', 'od': 'odisha',
    'or': 'odisha', 'pb': 'punjab', 'rj': 'rajasthan', 'sk': 'sikkim',
    'tn': 'tamil nadu', 'tg': 'telangana', 'ts': 'telangana', 'tr': 'tripura',
    'up': 'uttar pradesh', 'uk': 'uttarakhand', 'ut': 'uttarakhand', 'wb': 'west bengal',
    'dl': 'delhi', 'jk': 'jammu kashmir', 'la': 'ladakh', 'py': 'puducherry', 'ch': 'chandigarh'
}

INDIC_STATE_MAP = {
    'तमिलनाडु': 'tamil nadu', 'தமிழ்நாடு': 'tamil nadu', 'తమిళనాడు': 'tamil nadu', 'ತಮಿಳುನಾಡು': 'tamil nadu',
    'उत्तर प्रदेश': 'uttar pradesh', 'உத்தரப் பிரதேசம்': 'uttar pradesh', 'ఉత్తర ప్రదేశ్': 'uttar pradesh',
    'महाराष्ट्र': 'maharashtra', 'மகாராஷ்டிரா': 'maharashtra', 'మహారాష్ట్ర': 'maharashtra', 'મહારાષ્ટ્ર': 'maharashtra',
    'गुजरात': 'gujarat', 'குஜராத்': 'gujarat', 'గుజరాత్': 'gujarat', 'ગુજરાત': 'gujarat',
    'पश्चिम बंगाल': 'west bengal', 'மேற்கு வங்கம்': 'west bengal', 'পশ্চিমবঙ্গ': 'west bengal',
    'कर्नाटक': 'karnataka', 'கர்நாடகா': 'karnataka', 'కర్ణాటಕ': 'karnataka', 'ಕರ್ನಾಟಕ': 'karnataka',
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

LEET_MAP = {
    '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's',
    '6': 'g', '7': 't', '8': 'b'
}

def deleet_token(tok):
    if re.search(r'[a-zA-Z]', tok) and re.search(r'[0-9]', tok):
        return "".join(LEET_MAP.get(c, c) for c in tok)
    return tok

def deleet_text(text):
    if not text:
        return ""
    return " ".join(deleet_token(t) for t in text.split())

def strip_accents(text):
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def clean_text(text):
    if not text:
        return ""
    text = deleet_text(text)
    # Expand Indic state names first while Indic characters are pristine
    for k, v in INDIC_STATE_MAP.items():
        if k in text:
            text = text.replace(k, " " + v + " ")
    # Transliterate Indic characters to Roman phonetic equivalents
    text = transliterate_indic(text)
    # Strip European/French accents and combining marks on remaining Latin text
    text = strip_accents(text)
    text = text.lower()
    text = re.sub(r'\.(com|org|net|co|in|fr|io|info)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

FRENCH_ADDR_MAP = {
    r"\b(r|r\.)\b": "rue",
    r"\b(av|av\.)\b": "avenue",
    r"\b(bd|bd\.|bl)\b": "boulevard",
    r"\b(pl|pl\.)\b": "place",
    r"\b(all|all\.)\b": "allee",
    r"\b(imp|imp\.)\b": "impasse",
    r"\b(crs|crs\.)\b": "cours",
    r"\b(rte|rte\.)\b": "route",
    r"\b(ch|ch\.)\b": "chemin",
    r"\b(sq|sq\.)\b": "square",
    r"\b(pass|pass\.)\b": "passage"
}

def clean_address(text, country=None):
    if not text:
        return ""
    text = strip_accents(text).lower()
    if country == "France":
        for pat, repl in FRENCH_ADDR_MAP.items():
            text = re.sub(pat, " " + repl + " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def get_phonetic_skeleton(word):
    if len(word) < 2:
        return ""
    w = word.lower().replace("ph", "f").replace("ee", "i").replace("oo", "u")
    tr = str.maketrans({"g": "k", "d": "t", "b": "p", "v": "w", "z": "s", "j": "s", "c": "k", "q": "k", "x": "ks"})
    w = w.translate(tr)
    w = re.sub(r"(?<=[bcdfghjklmnpqrstvwxyz])h", "", w)
    sk = re.sub(r"[aeiouy\s]", "", w)
    sk = re.sub(r"(.)\1+", r"\1", sk)
    return sk if len(sk) >= 2 else ""

LEGAL_STOPWORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'llp', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'services', 'service', 'enterprises',
    'group', 'holdings', 'holding', 'associates', 'the', 'and', 'of', 'in', 'at',
    'sarl', 'sas', 'sasu', 'eurl', 'sci', 'sa', 'snc', 'gie', 'scp', 'sel', 'sem',
    'foundation', 'club', 'centre', 'center', 'international', 'global',
    'solutions', 'technologies', 'technology', 'to', 'for', 'by', 'on', 'with',
    'an', 'a', 'de', 'du', 'des', 'la', 'le', 'les', 'mr', 'mrs', 'ms', 'm', 'smt'
}

ADDR_STOPWORDS = {
    'street', 'st', 'road', 'rd', 'avenue', 'ave', 'drive', 'dr', 'lane', 'ln',
    'boulevard', 'blvd', 'way', 'court', 'ct', 'place', 'pl', 'highway', 'hwy',
    'floor', 'fl', 'suite', 'ste', 'room', 'rm', 'building', 'bldg', 'near',
    'opp', 'opposite', 'behind', 'post', 'box', 'po', 'rue', 'r', 'bd', 'av',
    'allée', 'allee', 'impasse', 'chemin', 'ch', 'cours', 'crs', 'rte', 'route',
    'residence', 'res', 'appartement', 'apt', 'north', 'south', 'east', 'west',
    'n', 's', 'e', 'w', 'null', 'none', 'bis', 'ter'
}

def clean_number(w):
    digits = re.sub(r'\D', '', w)
    return digits.lstrip('0') if digits else ""

def evaluate_entity_f05(gt_ids, pred_ids):
    """Evaluate F_0.5 score for a single Source 1 entity."""
    if len(gt_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0
    if len(pred_ids) == 0:
        return 0.0
    tp = len(gt_ids & pred_ids)
    if tp == 0:
        return 0.0
    p = tp / len(pred_ids)
    r = tp / len(gt_ids)
    denom = 0.25 * p + r
    return (1.25 * p * r) / denom if denom > 0 else 0.0

def evaluate_macro_f05(gt_dict, pred_dict):
    """Macro-averaged F_0.5 across all Source 1 entities."""
    scores = [evaluate_entity_f05(gt_dict[s1_id], pred_dict.get(s1_id, set())) for s1_id in gt_dict]
    return float(np.mean(scores))
