"""Time- and frequency-domain feature extraction for sensor windows."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FEATURE_NAMES = [
    "acc_mean_mag",
    "acc_var",
    "acc_std",
    "acc_rms",
    "acc_sma",
    "acc_corr_xy",
    "gyro_mean_mag",
    "gyro_var",
    "gyro_std",
    "dominant_freq_hz",
    "spectral_energy",
    "fft_peak_ratio",
]


def _sma(x: NDArray[np.floating], y: NDArray[np.floating], z: NDArray[np.floating]) -> float:
    return float(np.mean(np.abs(x) + np.abs(y) + np.abs(z)))


def _corr(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _dominant_freq(signal: NDArray[np.floating], fs: float) -> float:
    n = len(signal)
    if n < 4:
        return 0.0
    spec = np.abs(np.fft.rfft(signal - signal.mean()))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    if len(freqs) <= 1:
        return 0.0
    idx = int(np.argmax(spec[1:]) + 1)
    return float(freqs[idx])


def _spectral_energy(signal: NDArray[np.floating]) -> float:
    spec = np.fft.rfft(signal - signal.mean())
    return float(np.sum(np.abs(spec) ** 2) / len(signal))


def _fft_peak_ratio(signal: NDArray[np.floating]) -> float:
    spec = np.abs(np.fft.rfft(signal - signal.mean()))
    if spec.sum() < 1e-12:
        return 0.0
    return float(spec.max() / spec.sum())


def window_features(
    acc: NDArray[np.floating],
    gyro: NDArray[np.floating],
    fs: float,
) -> NDArray[np.float64]:
    """acc, gyro: shape (n_samples, 3)."""
    ax, ay, az = acc[:, 0], acc[:, 1], acc[:, 2]
    gx, gy, gz = gyro[:, 0], gyro[:, 1], gyro[:, 2]
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    feats = np.array(
        [
            acc_mag.mean(),
            acc_mag.var(),
            acc_mag.std(),
            float(np.sqrt(np.mean(acc_mag ** 2))),
            _sma(ax, ay, az),
            _corr(ax, ay),
            gyro_mag.mean(),
            gyro_mag.var(),
            gyro_mag.std(),
            _dominant_freq(acc_mag, fs),
            _spectral_energy(acc_mag),
            _fft_peak_ratio(acc_mag),
        ],
        dtype=np.float64,
    )
    return feats


def extract_sequence_features(
    df,
    window_samples: int,
    hop_samples: int,
    fs: float,
) -> tuple[NDArray[np.float64], list[tuple[float, float]]]:
    acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=float)
    gyro = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=float)
    t = df["seconds_elapsed"].to_numpy(dtype=float)

    vectors: list[NDArray[np.float64]] = []
    spans: list[tuple[float, float]] = []
    for start in range(0, len(df) - window_samples + 1, hop_samples):
        end = start + window_samples
        vectors.append(window_features(acc[start:end], gyro[start:end], fs))
        spans.append((float(t[start]), float(t[end - 1])))
    if not vectors:
        vectors.append(window_features(acc, gyro, fs))
        spans.append((float(t[0]), float(t[-1])))
    return np.vstack(vectors), spans


def fit_zscore(stats: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mean = stats.mean(axis=0)
    std = stats.std(axis=0)
    std[std < 1e-9] = 1.0
    return mean, std


def apply_zscore(x: NDArray[np.float64], mean: NDArray[np.float64], std: NDArray[np.float64]) -> NDArray[np.float64]:
    return (x - mean) / std
