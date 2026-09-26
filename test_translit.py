# A compact Brahmi phonetic mapping for Indic scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam)
# All Indic scripts share the same layout relative to their block start!
# Devanagari 0900, Bengali 0980, Gurmukhi 0A00, Gujarati 0A80, Oriya 0B00, Tamil 0B80, Telugu 0C00, Kannada 0C80, Malayalam 0D00

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
    0x4D: "", # virama (halant)
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

test_names = [
    "राम मार्केटिंग प्राइवेट लिमिटेड",
    "आदित्य प्रॉपर्टीज एलएलपी",
    "सन कंस्ट्रक्शंस प्राइवेट लिमिटेड",
    "रियल मॉडर्न फूड लिमिटेड",
    "குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்",
    "अल्फा अल टेक्नोलॉजीज प्राइवेट लिमिटेड",
    "ग्लोबल इन्वेस्टमेंट प्रा. लि.",
    "फर्स्ट फूड प्राइवेट लिमिटेड",
    "होटल वेंचर्स लिमिटेड",
    "जैन वेंचर्स प्राइवेट लिमिटेड"
]

for t in test_names:
    print(f"{t} -> {transliterate_indic(t)}")
