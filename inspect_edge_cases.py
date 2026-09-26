# Inspect the missing 14 pairs
import polars as pl
import unicodedata, re

from analyze_gt_pairs import normalize_text, get_tokens, s1_dict, s2_dict, s3_dict, matched_gt

missing = []
for r in matched_gt.iter_rows(named=True):
    s1_id = r["source1_entity_id"]
    r1 = s1_dict[s1_id]
    s1_name_toks = get_tokens(r1["business_name"])
    s1_addr_toks = get_tokens(r1["business_address"], remove_legal=False)

    for mid in r["matched_entity_ids"].split(","):
        r2 = s2_dict.get(mid) or s3_dict.get(mid)
        if not r2:
            continue
        r2_name_toks = get_tokens(r2["business_name"])
        r2_addr_toks = get_tokens(r2["business_address"], remove_legal=False) if r2["business_address"] else set()

        if not (s1_name_toks & r2_name_toks) and not (s1_addr_toks & r2_addr_toks):
            missing.append((r1, r2))

print(f"Total missing: {len(missing)}")
for r1, r2 in missing[:10]:
    print("---")
    print(f"S1: {r1['business_name']} | {r1['business_address']}")
    print(f"S2/3: {r2['business_name']} | {r2['business_address']}")
