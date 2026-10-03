import numpy as np
import os
import csv
from dotenv import load_dotenv

load_dotenv("./config.env")

DATASET = os.getenv("DATASET")
file_path = f"./{DATASET}/gloss_dict.npy"

data = np.load(file_path, allow_pickle=True).item()
data = sorted(data.items(), key=lambda d: d[1][1])

header = ["Gloss", "Frequency", "Index"]
csv_path = f"./{DATASET}/gloss_frequency.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for gloss, (idx, freq) in data:
        writer.writerow([gloss, freq, idx])
