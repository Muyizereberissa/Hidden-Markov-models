"""End-to-end training and evaluation pipeline for activity HMM."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from features import FEATURE_NAMES, apply_zscore, extract_sequence_features, fit_zscore
from hmm import GaussianHMM

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"

STATE_NAMES = ["Standing", "Walking", "Jumping", "Still"]
STATE_TO_IDX = {s: i for i, s in enumerate(STATE_NAMES)}

# At 100 Hz, 1 s windows (100 samples) capture roughly one gait cycle for walking/jumping.
WINDOW_SAMPLES = 100
HOP_SAMPLES = 50
FS = 100.0

# Fixed hold-out recordings for unseen evaluation (not used in Baum-Welch training).
TEST_FILES = [
    "Standing10-2026-10-28_18-54-27",
    "Walking19-2026-10-29_17-27-45",
    "Jumping19-2026-10-29_17-21-17",
    "Still20-2026-10-28_19-39-51",
]


def load_recording(stem: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{stem}.csv"
    if not path.exists():
        path = PROCESSED_DIR / f"{stem.replace('-', '_')}.csv"
    return pd.read_csv(path)


def file_stem(path: Path) -> str:
    return path.stem


def build_labeled_windows(csv_files: list[Path]) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    sequences: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    names: list[str] = []
    for path in csv_files:
        df = load_recording(file_stem(path))
        feats, _ = extract_sequence_features(df, WINDOW_SAMPLES, HOP_SAMPLES, FS)
        act = df["activity"].iloc[0]
        y = np.full(len(feats), STATE_TO_IDX[act], dtype=int)
        sequences.append(feats)
        labels.append(y)
        names.append(path.name)
    return sequences, labels, names


def set_supervised_emissions(model: GaussianHMM, feats: np.ndarray, labels: np.ndarray) -> None:
    for s in range(model.n_states):
        mask = labels == s
        if mask.sum() == 0:
            continue
        subset = feats[mask]
        model.means[s] = subset.mean(axis=0)
        model.vars[s] = np.maximum(subset.var(axis=0), 1e-3)


def initialize_hmm_from_labels(all_feats: np.ndarray, all_labels: np.ndarray, n_states: int, n_features: int) -> GaussianHMM:
    model = GaussianHMM(n_states, n_features)
    set_supervised_emissions(model, all_feats, all_labels)
    model.pi = np.ones(n_states) / n_states
    model.A = np.full((n_states, n_states), 0.03 / (n_states - 1))
    np.fill_diagonal(model.A, 0.91)
    return model


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> pd.DataFrame:
    rows = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        fp = np.sum((y_true != c) & (y_pred == c))
        tn = np.sum((y_true != c) & (y_pred != c))
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        acc = (tp + tn) / len(y_true) if len(y_true) else 0.0
        rows.append(
            {
                "State (Activity)": STATE_NAMES[c],
                "Number of Samples": int(np.sum(y_true == c)),
                "Sensitivity": round(sens, 4),
                "Specificity": round(spec, 4),
                "Overall Accuracy": round(acc, 4),
            }
        )
    return pd.DataFrame(rows)


def plot_transition_matrix(A: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(A, annot=True, fmt=".2f", xticklabels=STATE_NAMES, yticklabels=STATE_NAMES, cmap="Blues")
    plt.title("HMM Transition Matrix")
    plt.xlabel("To state")
    plt.ylabel("From state")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_emissions(model: GaussianHMM, out_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    sns.heatmap(model.means, cmap="viridis", xticklabels=FEATURE_NAMES, yticklabels=STATE_NAMES)
    plt.title("Emission Means (Gaussian per state)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_decoded_sequence(stem: str, out_path: Path, model: GaussianHMM, mean: np.ndarray, std: np.ndarray) -> None:
    df = load_recording(stem)
    feats, spans = extract_sequence_features(df, WINDOW_SAMPLES, HOP_SAMPLES, FS)
    obs = apply_zscore(feats, mean, std)
    path = model.predict(obs)
    t_mid = [(a + b) / 2 for a, b in spans]
    true_state = STATE_TO_IDX[df["activity"].iloc[0]]

    plt.figure(figsize=(10, 3))
    plt.step(t_mid, path, where="mid", label="Viterbi decoded")
    plt.axhline(true_state, color="red", linestyle="--", alpha=0.5, label="True activity index")
    plt.yticks(range(len(STATE_NAMES)), STATE_NAMES)
    plt.xlabel("Time (s)")
    plt.ylabel("State")
    plt.title(f"Decoded sequence: {stem}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_raw_sample(out_path: Path) -> None:
    sample = load_recording("Walking1-2026-10-28_19-09-39")
    if sample.empty:
        files = sorted(PROCESSED_DIR.glob("Walking*.csv"))
        sample = pd.read_csv(files[0])
    t = sample["seconds_elapsed"]
    plt.figure(figsize=(10, 4))
    plt.plot(t, sample["acc_x"], label="acc_x")
    plt.plot(t, sample["acc_y"], label="acc_y")
    plt.plot(t, sample["acc_z"], label="acc_z")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Sample accelerometer recording (Walking)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run() -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    all_csv = sorted(p for p in PROCESSED_DIR.glob("*.csv") if p.name != "manifest.csv")
    test_stems = {f"{s}.csv" for s in TEST_FILES}
    train_files = [p for p in all_csv if p.name not in test_stems]
    test_files = [p for p in all_csv if p.name in test_stems]

    train_seq, train_labels, _ = build_labeled_windows(train_files)
    test_seq, test_labels, test_names = build_labeled_windows(test_files)

    train_feats = np.vstack(train_seq)
    train_y = np.concatenate(train_labels)

    mean, std = fit_zscore(train_feats)
    train_norm = [apply_zscore(s, mean, std) for s in train_seq]
    test_norm = [apply_zscore(s, mean, std) for s in test_seq]

    train_norm_stack = np.vstack(train_norm)
    model = initialize_hmm_from_labels(train_norm_stack, train_y, len(STATE_NAMES), train_norm_stack.shape[1])
    log_liks = model.fit(train_norm, max_iter=100, tol=1e-4)
    # Keep class-specific emissions aligned with labeled windows after transition learning.
    set_supervised_emissions(model, train_norm_stack, train_y)

    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    for obs, y, name in zip(test_norm, test_labels, test_names):
        pred = model.predict(obs)
        y_true_all.extend(y.tolist())
        y_pred_all.extend(pred.tolist())

    y_true = np.array(y_true_all, dtype=int)
    y_pred = np.array(y_pred_all, dtype=int)
    metrics = per_class_metrics(y_true, y_pred, len(STATE_NAMES))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(STATE_NAMES))))

    plot_transition_matrix(model.A, FIGURES_DIR / "transition_matrix.png")
    plot_emissions(model, FIGURES_DIR / "emission_means.png")
    plot_raw_sample(FIGURES_DIR / "raw_accelerometer_sample.png")
    for stem in TEST_FILES:
        plot_decoded_sequence(stem, FIGURES_DIR / f"decoded_{stem}.png", model, mean, std)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=STATE_NAMES, yticklabels=STATE_NAMES, cmap="Greens")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix (unseen test recordings)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    metrics.to_csv(ROOT / "report" / "evaluation_metrics.csv", index=False)
    pd.DataFrame({"iteration": range(1, len(log_liks) + 1), "log_likelihood": log_liks}).to_csv(
        ROOT / "report" / "baum_welch_loglik.csv", index=False
    )

    return {
        "model": model,
        "metrics": metrics,
        "log_liks": log_liks,
        "mean": mean,
        "std": std,
        "n_train": len(train_files),
        "n_test": len(test_files),
    }


if __name__ == "__main__":
    result = run()
    print(result["metrics"].to_string(index=False))
    print(f"Training files: {result['n_train']}, test files: {result['n_test']}")
