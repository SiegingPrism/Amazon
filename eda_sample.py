import polars as pl
import pandas as pd

# Load first 20 rows of train_source1, train_source2, train_source3, train_ground_truth
s1 = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", n_rows=15)
s2 = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", n_rows=15)
s3 = pl.read_csv("dataset/train/train_source3.tsv", separator="\t", n_rows=15)
gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", n_rows=15)

print("--- Source 1 Sample ---")
print(s1)
print("\n--- Source 2 Sample ---")
print(s2)
print("\n--- Source 3 Sample ---")
print(s3)
print("\n--- Ground Truth Sample ---")
print(gt)
