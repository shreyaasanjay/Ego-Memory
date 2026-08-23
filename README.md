# EgoMemory

EgoMemory is a multimodal episodic-memory system for retrieving relevant moments from first-person cooking recordings. It tests whether visual, temporal, audio, and wearable-sensor signals improve retrieval over vision alone.

## Project question

**How much does each added modality improve event retrieval from egocentric experience?**

Given a question such as *“What happened immediately before I put the pan on the stove?”*, EgoMemory searches timestamped memory windows and returns the most relevant moments.

## Architecture

```text
Ego-Exo4D cooking take
  ├─ ego video ──────── visual embeddings ─┐
  ├─ audio ──────────── audio features ────┤
  ├─ IMU / trajectory ─ motion/location ───┤
  └─ narrations ─────── event metadata ────┘
                               ↓
                    timestamped memory store
                               ↓
                   FAISS visual search + reranking
                               ↓
                     top-k retrieved event windows
```

## What is implemented

- Safe project layout and ignored local dataset/results directories.
- A timestamp-window generator and downloaded-take modality inspector.
- A FAISS visual index with an automatic NumPy fallback.
- Temporal, location, audio-activity, and motion-feature reranking hooks.
- Recall@K and temporal-localization-error metrics.
- A data-free smoke test to verify retrieval plumbing before downloading data.

## Setup

Use **Python 3.11 (64-bit)**. Create a fresh environment; do not reuse an environment whose Python executable was removed.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[media,models,dev]"
pip install ego4d awscli
```


## Download exactly one cooking take

Ego-Exo4D access credentials are separate from AWS promotional credits. Configure only the access key issued with the dataset license; it is stored under your user profile, never in this repository.

```powershell
aws configure
egoexo --help
```

First download only metadata and annotations, use the visualizer/metadata to identify one cooking `UID`, then download that UID only:

```powershell
egoexo -o data/egoexo --parts metadata annotations --release v2
egoexo -o data/egoexo --uids <COOKING_TAKE_UID> --views ego --parts downscaled_takes/448 take_trajectory take_vrs_noimagestream --release v2
```

The second command intentionally requests a 448px ego video plus per-take trajectory and VRS-without-image-stream data. Confirm the exact part names with `egoexo --help` before accepting the downloader’s size estimate.

Inspect the downloaded take:

```powershell
python scripts/inspect_take.py data/egoexo/<TAKE_DIRECTORY>
```

## Experiment

Evaluate the same approximately 30 labeled queries using these configurations:

| System | Signals |
| --- | --- |
| Vision | visual embeddings |
| Vision + time | visual embeddings and temporal relationships |
| Vision + audio | visual, temporal, audio features |
| Full | visual, temporal, audio, trajectory/IMU features |

Report Recall@5 and temporal localization error. Do not make up results: populate the results table only after the selected cooking take and query labels have been processed.

## Repository layout

```text
src/egomemory/
  preprocessing/  # take inspection and synchronized windows
  retrieval/      # FAISS search and multimodal reranking
  evaluation/     # Recall@K and temporal error
configs/           # retrieval settings
scripts/           # inspection and smoke-test entry points
data/              # ignored restricted dataset files
results/           # ignored generated experiment artifacts
```

## Limitations

This initial version is intentionally a retrieval baseline, not a custom trained model or real-time system. Audio and sensor signals must be evaluated only where the selected take actually supplies synchronized recordings. Dataset-license restrictions also mean raw data cannot be committed to this repository.

## Sources

- [Ego-Exo4D downloader documentation](https://docs.ego-exo4d-data.org/download/)
- [Ego4D / Ego-Exo4D project repository](https://github.com/facebookresearch/Ego4d)
