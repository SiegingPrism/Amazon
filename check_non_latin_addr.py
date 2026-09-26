import polars as pl
import re

def script_type(text):
    if not text:
        return "empty"
    for s, name in [
        (r'[\u0900-\u097F]', "devanagari"),
        (r'[\u0B80-\u0BFF]', "tamil"),
        (r'[\u0C00-\u0C7F]', "telugu"),
        (r'[\u0C80-\u0CFF]', "kannada"),
        (r'[\u0980-\u09FF]', "bengali")
    ]:
        if re.search(s, text):
            return name
    return "latin"

s2_in = pl.read_csv("dataset/train/train_source2.tsv", separator="\t").filter(pl.col("country") == "India")

# Look at records where business_name is devanagari
dev_records = []
for r in s2_in.head(1000).iter_rows(named=True):
    if script_type(r["business_name"]) != "latin":
        dev_records.append(r)
    if len(dev_records) >= 10:
        break

for r in dev_records:
    print("Name:   ", r["business_name"])
    print("Address:", r["business_address"])
    print("Address script:", script_type(r["business_address"]))
    print()
