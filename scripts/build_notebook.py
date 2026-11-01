#!/usr/bin/env python3
"""Generate the main project notebook from repository modules."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebook" / "hmm_activity_recognition_full.ipynb"

cells = []


def md(source: str):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)})


def code(source: str):
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": source.splitlines(keepends=True),
        }
    )


md(
    """> **NOTEBOOK VERSION 3 (plots)** — Plot cells use `show_fig(fig)`, **not** `plt.show()`. Run the **setup cell** (with the crimson test plot) before other cells.

# Hidden Markov Model for Human Activity Recognition

This notebook documents data preparation, feature extraction, a **NumPy Gaussian HMM** (`src/hmm.py`), **Baum–Welch** training, **Viterbi** decoding, and evaluation on held-out recordings. Figures and tables here can be copied into the written report.
"""
)

md("""## 1. Setup and paths
""")

code(
    """import io
import inspect
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image, display


def show_fig(fig=None):
    # PNG bytes render reliably in Cursor/VS Code notebooks (plt.show() often does not).
    fig = fig or plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    display(Image(data=buf.getvalue()))


ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from features import (
    FEATURE_NAMES,
    apply_zscore,
    extract_sequence_features,
    fit_zscore,
    window_features,
)
from hmm import GaussianHMM
from pipeline import (
    FS,
    HOP_SAMPLES,
    PROCESSED_DIR,
    STATE_NAMES,
    TEST_FILES,
    WINDOW_SAMPLES,
    build_labeled_windows,
    plot_decoded_sequence,
    run,
)

sns.set_theme(style="whitegrid")

# Quick display test — a small plot should appear below this cell.
_test_fig, _test_ax = plt.subplots(figsize=(3, 1.2))
_test_ax.plot([0, 1], [0, 1], color="crimson", linewidth=2)
_test_ax.set_title("If you see this line, plot output works")
show_fig(_test_fig)
"""
)

md(
    """## 2. Data collection summary (for report)

| Activity | Duration | Recording note |
|----------|----------|----------------|
| Standing | 5–10 s | Phone steady at waist |
| Walking | 5–10 s | Steady indoor pace |
| Jumping | 5–10 s | Continuous jumps |
| Still | 5–10 s | Phone on flat surface |

**App:** Sensor Logger (Android). **Sensors:** accelerometer + gyroscope (x, y, z) with timestamps.

**Sampling rate:** Each export stores `native_sampling_rate_hz` in the processed CSV (from Sensor Logger metadata, typically **100 Hz** when `sampleRateMs = 10`). All clips were harmonized to **100 Hz** before windowing so window lengths stay consistent.
"""
)

code(
    """manifest = pd.read_csv(PROCESSED_DIR / "manifest.csv")
summary = manifest.groupby("activity").agg(
    files=("file", "count"),
    total_seconds=("duration_sec", "sum"),
    native_hz=("native_hz", "first"),
    harmonized_hz=("harmonized_hz", "first"),
)
summary
"""
)

md(
    """### 2.1 Pre-processing and harmonization

Raw Sensor Logger zips were merged (accelerometer + gyroscope on `seconds_elapsed`) and linearly interpolated to a uniform grid (`scripts/prepare_dataset.py`). Processed files live in `data/processed/` with an `activity` label in every row.
"""
)

code(
    """example = pd.read_csv(PROCESSED_DIR / "Jumping1-2026-10-29_17-05-49.csv")
example[["seconds_elapsed", "acc_x", "gyro_x", "activity", "native_sampling_rate_hz", "harmonized_sampling_rate_hz"]].head()
"""
)

md(
    """### 2.2 Raw sensor visualizations (include in report)

The plots below show typical traces per activity. Periodic motion (walking, jumping) appears in accelerometer amplitude; still vs standing differ mainly in variance and gyro activity.
"""
)

code(
    """sample_files = {
    "Standing": "Standing2-2026-10-28_18-39-54.csv",
    "Walking": "Walking1-2026-10-28_19-09-39.csv",
    "Jumping": "Jumping1-2026-10-29_17-05-49.csv",
    "Still": "Still3-2026-10-28_19-25-13.csv",
}

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for ax, (act, fname) in zip(axes.ravel(), sample_files.items()):
    df = pd.read_csv(PROCESSED_DIR / fname)
    ax.plot(df["seconds_elapsed"], df["acc_x"], alpha=0.8, label="acc_x")
    ax.plot(df["seconds_elapsed"], df["acc_y"], alpha=0.8, label="acc_y")
    ax.plot(df["seconds_elapsed"], df["acc_z"], alpha=0.8, label="acc_z")
    ax.set_title(act)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration")
ax.legend(loc="upper right", ncol=3, fontsize=8)
fig.suptitle("Raw accelerometer samples (one clip per activity)", y=1.02)
plt.tight_layout()
show_fig(fig)
"""
)

md(
    """### 2.3 Window size vs sampling rate

With `FS = 100` Hz, **window = 100 samples (1.0 s)** and **hop = 50 samples (0.5 s)**.

- At 100 Hz, one second ≈ one walking stride cycle and enough points for a stable FFT on jumping.
- A 5–10 s clip yields about **9–19** feature vectors—enough for a short HMM sequence without averaging away transitions.

*(Use this paragraph in the report “Data collection and pre-processing” section.)*
"""
)

md(
    """## 3. Feature extraction

### 3.1 Features and justification (report text)

| Feature | Domain | Why it helps |
|---------|--------|--------------|
| acc mean / var / std / RMS | Time | Separates intense motion (jump, walk) from low motion (still, stand) |
| acc SMA | Time | Captures overall movement intensity across axes |
| acc corr(x,y) | Time | Orientation/posture differs when phone is held vs flat on table |
| gyro mean / var / std | Time | Rotational motion is strong for jumping and walking |
| dominant FFT frequency | Frequency | Walking/jumping show periodic peaks; still is flat |
| spectral energy | Frequency | Total oscillation power distinguishes active vs inactive |
| FFT peak ratio | Frequency | Highlights narrow periodic motion vs broadband noise |

**Normalization:** **Z-score** per feature using **training-set only** statistics so high-scale FFT terms do not dominate Gaussian emissions.
"""
)

code("""pd.DataFrame({"feature": FEATURE_NAMES})
""")

code(
    """# Demo: one 1 s window from a walking clip
walk = pd.read_csv(PROCESSED_DIR / "Walking1-2026-10-28_19-09-39.csv")
acc = walk[["acc_x", "acc_y", "acc_z"]].to_numpy()
gyro = walk[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
single = window_features(acc[:WINDOW_SAMPLES], gyro[:WINDOW_SAMPLES], FS)
pd.DataFrame({"feature": FEATURE_NAMES, "value": single})
"""
)

code(
    """feats, spans = extract_sequence_features(walk, WINDOW_SAMPLES, HOP_SAMPLES, FS)
feat_df = pd.DataFrame(feats, columns=FEATURE_NAMES)
feat_df["t_start"] = [s[0] for s in spans]
feat_df["t_end"] = [s[1] for s in spans]
feat_df.head()
"""
)

code(
    """# Z-score example (fit on training data inside pipeline.run())
train_csv = sorted(p for p in PROCESSED_DIR.glob("*.csv") if p.name != "manifest.csv")
train_csv = [p for p in train_csv if p.stem not in TEST_FILES][:10]
seq, _, _ = build_labeled_windows(train_csv)
stack = np.vstack(seq)
mean, std = fit_zscore(stack)
normalized_head = apply_zscore(feats, mean, std)
pd.DataFrame(normalized_head, columns=FEATURE_NAMES).head()
"""
)

md(
    """## 4. HMM definition

| Symbol | Role in this project |
|--------|----------------------|
| **Z** | Hidden activity: Standing, Walking, Jumping, Still |
| **X** | 12-D Z-scored feature vector per window |
| **A** | Transition matrix \(P(z_t \\mid z_{t-1})\) |
| **B** | Diagonal Gaussian emissions \(P(x_t \\mid z_t)\) |
| **π** | Initial state distribution |

### Algorithms (brief, for report)

- **Baum–Welch (EM):** learns **A**, **B**, and **π** by maximizing sequence log-likelihood; we stop when the improvement in total log-likelihood is below **1e-4**.
- **Viterbi:** decodes the most likely state path—used for **sensitivity**, **specificity**, confusion matrix, and transition plots.

Implementation: `GaussianHMM` in `src/hmm.py` (NumPy only). Core methods:
"""
)

code(
    """print("=== Viterbi (decode) ===")
print(inspect.getsource(GaussianHMM.predict))
print("=== Baum–Welch (fit) ===")
print(inspect.getsource(GaussianHMM.fit))
"""
)

md("""## 5. Training, decoding, and unseen test data

**Unseen data:** four recordings (one per activity) listed in `TEST_FILES`—excluded before Baum–Welch. Describe in the report *how* you collected them (new session / same device) and whether the model generalizes.

**Training:** remaining files; emissions re-anchored to supervised class statistics after EM (`set_supervised_emissions`).
""")

code(
    """print("Held-out test files (unseen):")
for stem in TEST_FILES:
    print(" -", stem)
"""
)

code(
    """results = run()
metrics = results["metrics"]
model = results["model"]
metrics
"""
)

code(
    """print("Initial state π:")
print(pd.Series(model.pi, index=STATE_NAMES))
print("\\nTransition matrix A:")
pd.DataFrame(model.A, index=STATE_NAMES, columns=STATE_NAMES)
"""
)

code(
    """lik = pd.read_csv(ROOT / "report" / "baum_welch_loglik.csv")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(lik["iteration"], lik["log_likelihood"], marker="o")
ax.set_xlabel("Baum–Welch iteration")
ax.set_ylabel("Total log-likelihood")
ax.set_title("EM convergence (stopping when Δ log-likelihood < 1e-4)")
plt.tight_layout()
show_fig(fig)
"""
)

md("""## 6. Results and visualizations (for report figures)
""")

code(
    """fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(model.A, annot=True, fmt=".2f", xticklabels=STATE_NAMES, yticklabels=STATE_NAMES, cmap="Blues", ax=ax)
ax.set_title("Transition matrix heatmap")
ax.set_xlabel("To state")
ax.set_ylabel("From state")
plt.tight_layout()
show_fig(fig)
"""
)

code(
    """fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(model.means, cmap="viridis", xticklabels=FEATURE_NAMES, yticklabels=STATE_NAMES, ax=ax)
ax.set_title("Emission means (Gaussian B per state)")
plt.tight_layout()
show_fig(fig)

fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(model.vars, cmap="magma", xticklabels=FEATURE_NAMES, yticklabels=STATE_NAMES, ax=ax)
ax.set_title("Emission variances (diagonal covariance)")
plt.tight_layout()
show_fig(fig)
"""
)

code(
    """# Confusion matrix on held-out windows (regenerated in notebook)
from sklearn.metrics import confusion_matrix

all_csv = sorted(p for p in PROCESSED_DIR.glob("*.csv") if p.name != "manifest.csv")
test_files = [p for p in all_csv if p.stem in TEST_FILES]
test_seq, test_labels, _ = build_labeled_windows(test_files)
train_files = [p for p in all_csv if p.stem not in TEST_FILES]
train_seq, train_labels, _ = build_labeled_windows(train_files)
mean, std = fit_zscore(np.vstack(train_seq))
test_norm = [apply_zscore(s, mean, std) for s in test_seq]
y_true = np.concatenate(test_labels)
y_pred = np.concatenate([model.predict(obs) for obs in test_norm])
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(STATE_NAMES))))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=STATE_NAMES, yticklabels=STATE_NAMES, cmap="Greens", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion matrix (unseen test clips)")
plt.tight_layout()
show_fig(fig)
"""
)

code(
    """# Decoded Viterbi paths for all four held-out recordings
for stem in TEST_FILES:
    out = ROOT / "figures" / f"decoded_{stem}.png"
    plot_decoded_sequence(stem, out, model, mean, std)
    print("Saved", out.name)
"""
)

code(
    """fig_dir = ROOT / "figures"
for path in sorted(fig_dir.glob("decoded_*.png")):
    display(Image(data=path.read_bytes()))
"""
)

md(
    """## 7. Discussion prompts (copy into report)

**Easiest vs hardest activities:** Walking and jumping usually separate well via FFT and variance features. **Standing vs still** are both low-motion; gyro variance and acc axis correlation often decide between them.

**Transitions:** Large diagonal values in **A** mean activities persist across 0.5 s hops—expected for single-activity clips. Small off-diagonal values reflect noisy windows at boundaries.

**Noise and sampling rate:** Harmonizing to 100 Hz avoids mismatched window lengths. Sensor noise inflates variance features; Z-score normalization and 1 s windows reduce single-sample spikes.

**Improvements:** More participants and phones, person-wise validation splits, magnetometer/barometer features, longer multi-activity sessions to learn realistic **A**, hierarchical HMMs.

**Videos:** Add four 5 s links (face visible, full body + phone) in the report—required by the rubric.
"""
)

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "cells": cells,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {NB_PATH}")
