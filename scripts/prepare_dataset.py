#!/usr/bin/env python3
"""Extract Sensor Logger exports and build harmonized, labeled CSV recordings."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path(
    __import__("os").environ.get("HMM_SOURCE_DIR", "/Users/berissa/Documents/Hidden-Markov")
)
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TARGET_HZ = 100.0
ACTIVITIES = ("Standing", "Walking", "Jumping", "Still")


def parse_activity(name: str) -> str | None:
    for act in ACTIVITIES:
        if name.startswith(act):
            return act
    return None


def sample_rate_hz(metadata_path: Path) -> float:
    meta = pd.read_csv(metadata_path)
    if "sampleRateMs" not in meta.columns:
        return TARGET_HZ
    raw = str(meta["sampleRateMs"].iloc[0])
    ms = raw.split("|")[0].strip()
    try:
        interval_ms = float(ms)
        if interval_ms <= 0:
            return TARGET_HZ
        return 1000.0 / interval_ms
    except ValueError:
        return TARGET_HZ


def load_sensor_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values("seconds_elapsed").reset_index(drop=True)
    return df


def resample_sensor(df: pd.DataFrame, target_hz: float) -> pd.DataFrame:
    t = df["seconds_elapsed"].to_numpy(dtype=float)
    if len(t) < 2:
        return df.copy()
    duration = t[-1] - t[0]
    if duration <= 0:
        return df.copy()
    n = max(int(duration * target_hz) + 1, 2)
    grid = np.linspace(t[0], t[-1], n)
    out = {"seconds_elapsed": grid, "time": np.interp(grid, t, df["time"].to_numpy(dtype=float))}
    for col in ("x", "y", "z"):
        if col in df.columns:
            out[col] = np.interp(grid, t, df[col].to_numpy(dtype=float))
    return pd.DataFrame(out)


def merge_acc_gyro(acc: pd.DataFrame, gyro: pd.DataFrame) -> pd.DataFrame:
    acc = acc.rename(columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"})
    gyro = gyro.rename(columns={"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"})
    merged = pd.merge_asof(
        acc.sort_values("seconds_elapsed"),
        gyro.sort_values("seconds_elapsed"),
        on="seconds_elapsed",
        direction="nearest",
        suffixes=("_acc", "_gyro"),
    )
    time_col = "time_acc" if "time_acc" in merged.columns else "time"
    merged = merged.rename(columns={time_col: "timestamp"})
    keep = [
        "timestamp",
        "seconds_elapsed",
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    ]
    return merged[keep]


def process_export_folder(folder: Path, activity: str, out_name: str) -> Path | None:
    acc_path = folder / "Accelerometer.csv"
    gyro_path = folder / "Gyroscope.csv"
    meta_path = folder / "Metadata.csv"
    if not acc_path.exists() or not gyro_path.exists():
        return None

    native_hz = sample_rate_hz(meta_path) if meta_path.exists() else TARGET_HZ
    acc = resample_sensor(load_sensor_csv(acc_path), TARGET_HZ)
    gyro = resample_sensor(load_sensor_csv(gyro_path), TARGET_HZ)
    merged = merge_acc_gyro(acc, gyro)
    merged["activity"] = activity
    merged["native_sampling_rate_hz"] = native_hz
    merged["harmonized_sampling_rate_hz"] = TARGET_HZ

    out_path = PROCESSED_DIR / f"{out_name}.csv"
    merged.to_csv(out_path, index=False)
    return out_path


def extract_zip(zip_path: Path, dest: Path) -> Path | None:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
    except zipfile.BadZipFile:
        return None
    # Sensor Logger zips extract CSVs directly into dest
    if (dest / "Accelerometer.csv").exists():
        return dest
    subdirs = [p for p in dest.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and (subdirs[0] / "Accelerometer.csv").exists():
        return subdirs[0]
    return dest if (dest / "Accelerometer.csv").exists() else None


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    zips = sorted(SOURCE_DIR.glob("*.zip"))
    for zip_path in zips:
        activity = parse_activity(zip_path.stem)
        if activity is None:
            continue
        safe_stem = re.sub(r"[^\w\-]", "_", zip_path.stem)
        extract_dir = RAW_DIR / safe_stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        folder = extract_zip(zip_path, extract_dir)
        if folder is None:
            continue
        out = process_export_folder(folder, activity, safe_stem)
        if out is not None:
            written.append(out)

    # Also pick up already-extracted folders in source directory
    for folder in sorted(SOURCE_DIR.iterdir()):
        if not folder.is_dir():
            continue
        activity = parse_activity(folder.name)
        if activity is None:
            continue
        safe_stem = re.sub(r"[^\w\-]", "_", folder.name)
        out_path = PROCESSED_DIR / f"{safe_stem}.csv"
        if out_path.exists():
            continue
        out = process_export_folder(folder, activity, safe_stem)
        if out is not None:
            written.append(out)

    manifest = []
    for csv_path in sorted(PROCESSED_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path, nrows=5)
        full = pd.read_csv(csv_path)
        duration = full["seconds_elapsed"].iloc[-1] - full["seconds_elapsed"].iloc[0]
        manifest.append(
            {
                "file": csv_path.name,
                "activity": full["activity"].iloc[0],
                "rows": len(full),
                "duration_sec": round(float(duration), 3),
                "native_hz": float(full["native_sampling_rate_hz"].iloc[0]),
                "harmonized_hz": float(full["harmonized_sampling_rate_hz"].iloc[0]),
            }
        )

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(PROCESSED_DIR / "manifest.csv", index=False)
    print(f"Processed {len(written)} new files; {len(manifest_df)} total in {PROCESSED_DIR}")
    print(manifest_df.groupby("activity").agg(count=("file", "count"), total_sec=("duration_sec", "sum")))


if __name__ == "__main__":
    main()
