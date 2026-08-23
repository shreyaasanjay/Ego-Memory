# EgoMemory

EgoMemory is a multimodal episodic-memory system for retrieving relevant moments from video recordings. It tests whether visual, temporal, audio, and wearable-sensor signals improve retrieval over vision alone.

## Project question

**How much does each added modality improve event retrieval from egocentric experience?**

Given a question such as *“What happened immediately before I put the pan on the stove?”*, EgoMemory searches timestamped memory windows and returns the most relevant moments. Cooking is the demonstration domain—not a system limitation.

## Architecture

```text
Time-aligned video recording
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


## Local demo

After building the selected take's index, launch the lightweight demo:

```powershell
pip install -e ".[ui]"
python -m streamlit run app.py
```

On Windows, from the project folder you can instead run `run_demo.cmd`.

### Native desktop demo

For a local desktop window instead of the browser UI, run `run_desktop.cmd` from the project folder. It includes embedded playback of the selected retrieved window, modality scores, and the aligned transcript.

## Fixed-weight ablation results

Run the experiment with `python scripts/run_ablation.py`. It evaluates the labeled query set under vision, audio-transcript, motion, pairwise, and full-multimodal configurations. Outputs are written to `results/ablation/`; the desktop app displays the summary table. These results use predetermined, normalized modality weights—no per-query tuning.

To reproduce the balanced 50-query provisional run:

```powershell
python scripts/run_ablation.py --queries configs/evaluation_queries_50_provisional.json --output results/ablation_50_provisional
```

It lets you enter a question, select a retrieved event, play the ego video from that timestamp, and inspect the visual, audio-transcript, and motion evidence used for ranking.

## Download one Ego-Exo4D demonstration take

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

### Current balanced 50-query development run

This repository contains 10 vision, 10 audio, 10 motion, 10 vision+audio, and 10 full-multimodal prompts. The audio labels are grounded in the official timestamped transcription. The remaining labels are clearly marked **provisional** pending visual review.

| System | Recall@5 | Avg. temporal error |
| --- | ---: | ---: |
| Vision | 30.0% | 141.8 s |
| Audio/transcript | 74.0% | 39.0 s |
| Motion | 18.0% | 173.5 s |
| Vision + audio | 70.0% | 103.7 s |
| Full multimodal | 72.0% | 115.0 s |

These are development results, not final research claims. Re-label the provisional events after visual review, then rerun the experiment.

## Beyond cooking

EgoMemory's retrieval design works for any timestamped video archive: wearable-camera footage, meetings, lectures, sports practice, maintenance videos, travel footage, laboratory sessions, or security video.

- **Vision:** video frames sampled into time windows.
- **Audio:** soundtrack and/or timestamped transcription.
- **Time:** timestamps and neighboring event windows.
- **Motion/spatial context:** optional IMU, GPS, camera trajectory, or location metadata.

Sources without sensor data still work as video + audio + temporal retrieval systems; motion simply becomes unavailable rather than required.

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
