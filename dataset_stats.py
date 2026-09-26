import polars as pl

print("Reading train sources...")
s1 = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
s2 = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
s3 = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")
gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")

print(f"Train S1 count: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}, GT: {len(gt)}")
print("Train S1 country value counts:")
print(s1["country"].value_counts())
print("Train S2 country value counts:")
print(s2["country"].value_counts())
print("Train S3 country value counts:")
print(s3["country"].value_counts())

print("\nReading test sources...")
ts1 = pl.read_csv("dataset/test/test_source1.tsv", separator="\t")
ts2 = pl.read_csv("dataset/test/test_source2.tsv", separator="\t")
ts3 = pl.read_csv("dataset/test/test_source3.tsv", separator="\t")

print(f"Test S1 count: {len(ts1)}, S2: {len(ts2)}, S3: {len(ts3)}")
print("Test S1 country value counts:")
print(ts1["country"].value_counts())
print("Test S2 country value counts:")
print(ts2["country"].value_counts())
print("Test S3 country value counts:")
print(ts3["country"].value_counts())
