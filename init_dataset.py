import os
import pandas as pd
from test_blocking import make_synthetic

os.makedirs("dataset/train", exist_ok=True)
os.makedirs("dataset/test", exist_ok=True)

s1, s2, s3 = make_synthetic()

# Save for test
s1.to_csv("dataset/test/test_source1.tsv", sep="\t", index=False)
s2.to_csv("dataset/test/test_source2.tsv", sep="\t", index=False)
s3.to_csv("dataset/test/test_source3.tsv", sep="\t", index=False)

# Save for train
s1.to_csv("dataset/train/train_source1.tsv", sep="\t", index=False)
s2.to_csv("dataset/train/train_source2.tsv", sep="\t", index=False)
s3.to_csv("dataset/train/train_source3.tsv", sep="\t", index=False)

# Ground truth for train
gt = pd.DataFrame([
    {"source1_entity_id": "S1-732914", "matched_entity_ids": "S2-118820,S3-905477"},
    {"source1_entity_id": "S1-889301", "matched_entity_ids": "S2-397155,S3-651230"},
    {"source1_entity_id": "S1-999999", "matched_entity_ids": ""},
    {"source1_entity_id": "S1-FR0001", "matched_entity_ids": "S2-FR0002"},
])
gt.to_csv("dataset/train/train_ground_truth.tsv", sep="\t", index=False)
print("Synthetic datasets created successfully.")
