import polars as pl

ts1 = pl.read_csv("dataset/test/test_source1.tsv", separator="\t").filter(pl.col("country") == "France").head(10)
ts2 = pl.read_csv("dataset/test/test_source2.tsv", separator="\t").filter(pl.col("country") == "France").head(10)
ts3 = pl.read_csv("dataset/test/test_source3.tsv", separator="\t").filter(pl.col("country") == "France").head(10)

print("--- France Test S1 ---")
for r in ts1.iter_rows(named=True):
    print(r)

print("\n--- France Test S2 ---")
for r in ts2.iter_rows(named=True):
    print(r)

print("\n--- France Test S3 ---")
for r in ts3.iter_rows(named=True):
    print(r)
