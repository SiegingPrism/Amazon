import polars as pl
import re

def script_type(text):
    if not text:
        return "empty"
    # Devanagari range: \u0900-\u097F
    if re.search(r'[\u0900-\u097F]', text):
        return "devanagari"
    # Tamil range: \u0B80-\u0BFF
    if re.search(r'[\u0B80-\u0BFF]', text):
        return "tamil"
    # Telugu: \u0C00-\u0C7F
    if re.search(r'[\u0C00-\u0C7F]', text):
        return "telugu"
    # Kannada: \u0C80-\u0CFF
    if re.search(r'[\u0C80-\u0CFF]', text):
        return "kannada"
    # Bengali: \u0980-\u09FF
    if re.search(r'[\u0980-\u09FF]', text):
        return "bengali"
    return "latin"

s1_in = pl.read_csv("dataset/train/train_source1.tsv", separator="\t").filter(pl.col("country") == "India")
s2_in = pl.read_csv("dataset/train/train_source2.tsv", separator="\t").filter(pl.col("country") == "India")
s3_in = pl.read_csv("dataset/train/train_source3.tsv", separator="\t").filter(pl.col("country") == "India")

print("Train India S1 name script distribution (first 50k):")
s1_scripts = [script_type(x) for x in s1_in["business_name"].head(50000).to_list()]
print(pl.Series(s1_scripts).value_counts())

print("Train India S2 name script distribution (first 50k):")
s2_scripts = [script_type(x) for x in s2_in["business_name"].head(50000).to_list()]
print(pl.Series(s2_scripts).value_counts())

print("Train India S3 name script distribution (first 50k):")
s3_scripts = [script_type(x) for x in s3_in["business_name"].head(50000).to_list()]
print(pl.Series(s3_scripts).value_counts())
