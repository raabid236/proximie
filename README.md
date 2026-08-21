# Proximie Senior ML Engineer Challenge

Prototype for live Operating Room (OR) action segmentation under multi-camera, low-bandwidth, and strict data-retention constraints.

## What this repository demonstrates

- **Multi-sensor feature ingestion** using deterministic mock features.
- **Temporal classification** using a lightweight PyTorch GRU.
- **Timeline post-processing** that removes short noisy runs and merges adjacent segments.
- **Dual metric stack**:
  - model metrics: frame accuracy, macro F1, per-class F1
  - product metrics: segment IoU, boundary MAE, fragmentation rate, patient-present false-positive duration
- **Mock error analysis** focused on the two failure modes from the challenge:
  - `PATIENT_PRESENT`: long/static phase with false positives and fragmentation
  - `OPERATION`: occlusion-heavy phase with poor temporal boundaries
- **Reproducible configuration** and unit tests.
- **Architecture report** covering annotation lifecycle, active learning, multi-view fusion, temporal modeling, AWS deployment, and cost controls.

## Repository layout

```text
.
├── configs/default.yaml
├── docs/technical_architecture_report.md
├── docs/technical_architecture_report.pdf
├── scripts/run_pipeline.py
├── scripts/error_analysis.py
├── src/proximie_pipeline/
│   ├── data.py
│   ├── metrics.py
│   ├── model.py
│   ├── postprocess.py
│   └── pipeline.py
├── tests/
│   ├── test_metrics.py
│   └── test_postprocess.py
├── requirements.txt
└── pyproject.toml
```

## Quick start

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_pipeline.py
python scripts/error_analysis.py
pytest -q
```

The main pipeline prints model metrics and product-oriented segmentation metrics. The error-analysis script intentionally injects different noise patterns into `PATIENT_PRESENT` and `OPERATION` so the output resembles the expected failure analysis rather than claiming production accuracy.

## Design choices

### Feature ingestion

The mock generator emits a tensor shaped:

```text
[time, sensors, feature_dim]
```

Each sensor has a deterministic latent phase signal plus sensor-specific noise and an occlusion probability. In production, this interface would be fed by online visual embeddings rather than RGB frames.

### Temporal model

A small GRU consumes the concatenated multi-sensor embeddings:

```text
per-camera embeddings
        ↓
sensor mask / quality features
        ↓
concatenate
        ↓
2-layer GRU
        ↓
linear classifier
        ↓
frame-level phase logits
```

The prototype deliberately avoids a large vision backbone. The challenge is about temporal-system structure, reproducibility, and production reasoning.

### Post-processing

Frame predictions are converted into contiguous segments with:

1. median filtering,
2. minimum-duration removal,
3. neighboring-segment merging,
4. optional class-specific minimum duration.

This is intentionally simple. A production system could replace this with a semi-Markov decoder, HMM/CRF-style constraints, or learned boundary refinement.

## Expected phase labels

```text
0 = EMPTY
1 = PATIENT_PRESENT
2 = OPERATION
3 = POST_OPERATION
```

## Productionization path

The report proposes a streaming AWS architecture in which raw video is used only in-flight, while privacy-approved derived features, quality metadata, hard-example events, and short-lived debugging artifacts are retained according to governance rules. Historical raw video is handled by an offline curation plane with explicit TTLs and access controls.

## Reproducibility

The mock data generator accepts a seed. Configuration is stored in `configs/default.yaml`. The pipeline does not depend on the MM-OR dataset and runs without downloading external data.
