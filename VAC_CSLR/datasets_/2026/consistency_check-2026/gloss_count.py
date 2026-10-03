import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
from collections import Counter

# ── paths ──────────────────────────────────────────────────────────────────
file = "/draft-hdd-projects/captarlibras_finep/cauamagalhaes/preprocess_branch/captar-libras-recog-sinal/0_baselines/VAC_CSLR/ELAN/consistency_check-2026/consistencia_file.csv"
out_dir = Path(file).parent
df = pd.read_csv(file)
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nNull counts:\n{df.isnull().sum()}")

# ── gloss tokenization ─────────────────────────────────────────────────────
glosses = df["Gloss"].dropna().tolist()
all_tokens = [g for sentence in glosses for g in sentence.split()]
gloss_count = Counter(all_tokens)
gloss_set = set(gloss_count.keys())
sorted_count = dict(sorted(gloss_count.items(), key=lambda x: x[1], reverse=True))

print(f"\nNumber of unique glosses : {len(gloss_set)}")
print(f"Total gloss tokens       : {len(all_tokens)}")
print(f"Vocabulary size          : {len(gloss_set)}")
print(f"Avg glosses per sentence : {len(all_tokens) / len(glosses):.2f}")
print(f"Max glosses in sentence  : {max(len(s.split()) for s in glosses)}")
print(f"Min glosses in sentence  : {min(len(s.split()) for s in glosses)}")

sentence_lengths = [len(s.split()) for s in glosses]


# ── helper ─────────────────────────────────────────────────────────────────
def savefig(name):
    path = out_dir / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# 1. Gloss frequency bar chart (top-50)
# ══════════════════════════════════════════════════════════════════════════
top_n = 50
top_glosses = list(sorted_count.keys())[:top_n]
top_counts = [sorted_count[g] for g in top_glosses]

fig, ax = plt.subplots(figsize=(18, 5))
ax.bar(range(top_n), top_counts, color="steelblue", edgecolor="white", linewidth=0.4)
ax.set_xticks(range(top_n))
ax.set_xticklabels(top_glosses, rotation=90, fontsize=7)
ax.set_xlabel("Gloss")
ax.set_ylabel("Count")
ax.set_title(f"Top-{top_n} Most Frequent Glosses")
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
savefig("01_gloss_top50_bar.png")

# ══════════════════════════════════════════════════════════════════════════
# 2. Gloss frequency — full ranked plot (log scale)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(range(len(sorted_count)), list(sorted_count.values()), color="steelblue", lw=1.2)
ax.set_yscale("log")
ax.set_xlabel("Gloss rank")
ax.set_ylabel("Frequency (log)")
ax.set_title("Gloss Frequency Distribution (Zipf curve)")
ax.grid(True, which="both", ls="--", alpha=0.4)
savefig("02_gloss_zipf.png")

# ══════════════════════════════════════════════════════════════════════════
# 3. Hapax legomena / frequency buckets
# ══════════════════════════════════════════════════════════════════════════
buckets = {
    "1 (hapax)": sum(1 for v in gloss_count.values() if v == 1),
    "2–5": sum(1 for v in gloss_count.values() if 2 <= v <= 5),
    "6–20": sum(1 for v in gloss_count.values() if 6 <= v <= 20),
    "21–50": sum(1 for v in gloss_count.values() if 21 <= v <= 50),
    "51–100": sum(1 for v in gloss_count.values() if 51 <= v <= 100),
    ">100": sum(1 for v in gloss_count.values() if v > 100),
}
print(f"\nGloss frequency buckets:\n{buckets}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(buckets.keys(), buckets.values(), color="coral", edgecolor="white")
ax.set_xlabel("Frequency range")
ax.set_ylabel("Number of unique glosses")
ax.set_title("Gloss Vocabulary by Frequency Bucket")
for i, (k, v) in enumerate(buckets.items()):
    ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=9)
savefig("03_gloss_freq_buckets.png")

# ══════════════════════════════════════════════════════════════════════════
# 4. Sentence length histogram
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(
    sentence_lengths, bins=range(1, max(sentence_lengths) + 2), color="mediumseagreen", edgecolor="white", align="left"
)
ax.set_xlabel("Number of glosses per sentence")
ax.set_ylabel("Count")
ax.set_title("Sentence Length Distribution")
ax.axvline(np.mean(sentence_lengths), color="red", ls="--", lw=1.5, label=f"Mean = {np.mean(sentence_lengths):.1f}")
ax.axvline(
    np.median(sentence_lengths), color="orange", ls="--", lw=1.5, label=f"Median = {np.median(sentence_lengths):.1f}"
)
ax.legend()
savefig("04_sentence_length_hist.png")

# ══════════════════════════════════════════════════════════════════════════
# 5. Signer distribution  (if "Signer" / "signer_id" column exists)
# ══════════════════════════════════════════════════════════════════════════
signer_col = next((c for c in df.columns if "signer" in c.lower()), None)
if signer_col:
    signer_counts = df[signer_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(max(8, len(signer_counts) * 0.4), 4))
    signer_counts.plot(kind="bar", ax=ax, color="mediumpurple", edgecolor="white")
    ax.set_xlabel("Signer")
    ax.set_ylabel("Number of samples")
    ax.set_title("Samples per Signer")
    ax.tick_params(axis="x", rotation=45)
    savefig("05_signer_distribution.png")

# ══════════════════════════════════════════════════════════════════════════
# 6. Samples per split  (if "split" / "partition" column exists)
# ══════════════════════════════════════════════════════════════════════════
split_col = next((c for c in df.columns if c.lower() in {"split", "partition", "subset", "set"}), None)
if split_col:
    split_counts = df[split_col].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    split_counts.plot(kind="bar", ax=ax, color=["steelblue", "coral", "mediumseagreen"], edgecolor="white")
    ax.set_xlabel("Split")
    ax.set_ylabel("Count")
    ax.set_title("Samples per Split")
    ax.tick_params(axis="x", rotation=0)
    for i, v in enumerate(split_counts):
        ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=10)
    savefig("06_split_distribution.png")

# ══════════════════════════════════════════════════════════════════════════
# 7. Summary stats table — saved as CSV
# ══════════════════════════════════════════════════════════════════════════
summary = {
    "total_sentences": len(glosses),
    "total_gloss_tokens": len(all_tokens),
    "unique_glosses": len(gloss_set),
    "hapax_legomena": buckets["1 (hapax)"],
    "hapax_ratio": round(buckets["1 (hapax)"] / len(gloss_set), 4),
    "avg_sentence_length": round(np.mean(sentence_lengths), 2),
    "median_sentence_length": int(np.median(sentence_lengths)),
    "max_sentence_length": max(sentence_lengths),
    "min_sentence_length": min(sentence_lengths),
    "std_sentence_length": round(np.std(sentence_lengths), 2),
    "most_frequent_gloss": list(sorted_count.keys())[0],
    "most_frequent_count": list(sorted_count.values())[0],
}
summary_path = out_dir / "stats_summary.csv"
pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(summary_path)
print(f"\nSaved summary: {summary_path}")
print(pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_string())
