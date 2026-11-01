# Hidden Markov Models - Human Activity Recognition

Smartphone accelerometer and gyroscope recordings (Sensor Logger) for four activities: **Standing**, **Walking**, **Jumping**, and **Still**.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# If raw exports live outside the repo, update SOURCE_DIR in scripts/prepare_dataset.py, then:
python3 scripts/prepare_dataset.py

PYTHONPATH=src python3 src/pipeline.py
python3 scripts/build_notebook.py
jupyter notebook notebooks/hmm_activity_recognition.ipynb
```

## Contents

- `data/processed/` — 85 labeled CSV files + `manifest.csv`
- `notebooks/hmm_activity_recognition.ipynb` — feature extraction, HMM training, evaluation
- `src/hmm.py` — Gaussian HMM (Viterbi + Baum–Welch)
- `report/HMM_Activity_Recognition_Report.md` — draft written report (add video links)
- `figures/` — transition matrix, emissions, confusion matrix, decoded sequences

## Test hold-out

Four files (one per activity) are excluded from training; see `TEST_FILES` in `src/pipeline.py`.
