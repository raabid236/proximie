# Technical Architecture Report — Proximie Live OR Action Segmentation

**Proposed prototype:** multi-camera feature ingestion → quality-aware sensor fusion → temporal model → boundary-aware post-processing → dual ML/product evaluation.

## Executive summary

The core design principle is to separate **real-time clinical inference** from **data retention and annotation**. Raw video is treated as an ephemeral input to the streaming plane. The model consumes compact, privacy-approved visual embeddings plus camera-quality metadata, while a separate event stream records model uncertainty, disagreement, boundary instability, and sensor failures. These derived artifacts can support monitoring and active learning without creating a permanent copy of the raw feed.

The modeling stack uses three ideas that directly target the stated failures:

1. **Long-context temporal modeling** for the long, mostly static `PATIENT_PRESENT` phase.
2. **Short-context/high-resolution features and boundary refinement** for `OPERATION`.
3. **Quality-aware multi-view fusion**, so an occluded camera contributes less instead of corrupting the fused representation.

The prototype intentionally uses mock embeddings and a small GRU. Production scale should move the same interfaces to a streaming feature extractor and a more efficient multi-scale temporal architecture.

---

# 1. Data Strategy & Annotation Lifecycle

## 1.1 In-flight hard-example capture without retaining raw video

I would create a **derived-event side channel** next to the live inference path.

```text
camera streams
     │
     ▼
hospital gateway
     │
     ▼
AWS video ingestion
     │
     ├──────────────► ephemeral frame decode
     │                         │
     │                         ▼
     │                  visual embedding
     │                         │
     │                         ▼
     │                  feature synchronizer
     │                         │
     │                         ▼
     │                  temporal inference
     │                         │
     │              ┌──────────┴──────────┐
     │              ▼                     ▼
     │        workflow timeline       quality/hardness
     │                                    │
     │                                    ▼
     │                         derived event / feature
     ▼
raw-video path expires according to policy
```

The model should emit a compact **hard-example record** containing, for example:

- OR/session pseudonymous identifier
- timestamp bucket rather than patient identity
- predicted class probabilities
- top-2 class margin
- temporal entropy
- camera-quality scores
- inter-camera disagreement
- predicted boundary confidence
- model/version/configuration identifiers
- feature-vector reference if governance allows feature retention

Examples should be selected when:

- confidence is low;
- two classes repeatedly alternate;
- cameras disagree;
- the boundary head is uncertain;
- a rare class is detected;
- `PATIENT_PRESENT` appears for an unusually short interval;
- `OPERATION` contains repeated quality dropouts;
- a production rule detects an implausible workflow transition.

The important governance boundary is that **feature retention is not automatically equivalent to video retention**. Embeddings can still be sensitive data, so they should be classified, encrypted, access-controlled, TTL-managed, and assessed for reversibility/inference risk. If the privacy agreement does not permit persistent embeddings, retain only aggregate statistics and a short-lived review token.

I would explicitly avoid creating a hidden "debug video bucket." If a raw clip is required for an approved investigation, it should be an exception workflow with authorization, short TTL, encryption, audit logging, and automatic deletion.

## 1.2 Active learning over historical terabytes

I would build a batch curation plane rather than ask annotators to randomly sample the archive.

### Stage A — cheap candidate generation

Run an inexpensive embedding model over historical streams. Store compact representations and metadata rather than copying every frame into an annotation dataset.

For every temporal window, calculate:

- embedding diversity;
- uncertainty;
- temporal instability;
- camera disagreement;
- phase-transition likelihood;
- motion/scene-change statistics;
- frequency of quality failures;
- rarity relative to the current labeled set.

### Stage B — diversity-aware selection

Pure uncertainty sampling tends to select many nearly identical frames from the same failure mode. I would therefore combine:

**informativeness × diversity × coverage**

A practical implementation is:

1. score windows by uncertainty / disagreement;
2. cluster candidates in embedding space;
3. select representatives with farthest-point sampling or k-center greedy;
4. enforce quotas for rare classes, hospitals, camera configurations, and workflow positions.

This makes an annotation batch useful instead of merely large.

### Stage C — temporal clips, not isolated frames

For action segmentation, annotators should receive short clips around candidate transitions. If a model flags an uncertain boundary at time `t`, sample `[t - Δ, t + Δ]` rather than one frame.

The active-learning unit should therefore be a **temporal clip with context**, while annotation can produce frame ranges or boundary timestamps.

## 1.3 Annotation optimization

I would use a three-stage loop:

**self-supervised pre-training → teacher pseudo-labels → targeted human correction**

A video/feature encoder can first learn temporal invariances from abundant unlabeled streams using masked-feature prediction, temporal contrastive learning, or future-step prediction.

Then:

1. train an initial supervised model on a small seed set;
2. generate pseudo-labels on unlabeled data;
3. automatically accept only high-confidence, temporally consistent pseudo-labels;
4. send uncertain boundaries and disagreement cases to humans;
5. retrain;
6. repeat.

For segmentation, I would prioritize annotation of **boundaries** because a single temporal label can improve an entire interval. An annotation UI should allow fast boundary dragging, segment merging/splitting, and keyboard shortcuts.

A useful cost metric is not only annotation hours but **model improvement per annotation minute**.

---

# 2. Modeling & Multi-Sensor Fusion

## 2.1 Multi-view spatial fusion

The sensors are fixed in the room, which is a major advantage. During installation, I would calibrate each camera using:

- intrinsic calibration;
- extrinsic pose;
- a shared OR coordinate system;
- timestamp synchronization metadata.

The first production representation would be a quality-aware feature fusion layer rather than pixel-level fusion.

For synchronized time `t`, each camera produces:

`e_i(t) = encoder(frame_i(t))`

and a quality score:

`q_i(t) = visibility / blur / exposure / occlusion confidence`.

Then:

`e_fused(t) = Σ_i w_i e_i(t) / Σ_i w_i`

where `w_i` is derived from `q_i`.

The fused embedding is accompanied by:

- mean camera quality;
- minimum camera quality;
- camera disagreement;
- active-camera count.

This gives the temporal model an explicit indication that a fused representation is unreliable.

For more advanced production modeling, I would use **cross-view attention**. Each camera becomes a token; the model learns to select the best view at each timestep. Fixed-camera geometry can also provide relative-position embeddings so the model knows which viewpoint produced each token.

Pixel-level geometric fusion is valuable when calibration is reliable and the same physical regions can be mapped to a common coordinate system, but it is more brittle under surgical occlusion and changing equipment. Feature-level fusion is therefore my initial production choice.

## 2.2 Temporal scale balancing

The two failure modes imply two temporal scales.

`PATIENT_PRESENT` needs a long receptive field because isolated movement should not create a new phase.

`OPERATION` needs high temporal resolution because start/end boundaries matter and occlusions can create short gaps.

I would use a **multi-scale temporal encoder**:

```text
camera embeddings
       │
       ▼
quality-aware fusion
       │
       ├──────────────► 1–2 Hz high-resolution stream
       │                     │
       │                     ▼
       │               short TCN / local attention
       │                     │
       └──────────────► downsampled 0.25 Hz stream
                             │
                             ▼
                        long-context TCN
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
           phase classifier         boundary head
```

The high-resolution branch handles rapid transitions. The low-resolution branch provides several minutes of context at modest compute cost. Their representations are fused before classification.

For production, I would favor dilated temporal convolutions or efficient local attention over a very large Transformer. They provide predictable memory usage and can support streaming inference.

I would also add a **boundary head** predicting start/end probability. This directly optimizes the metric the product cares about rather than hoping frame classification alone learns precise boundaries.

For `PATIENT_PRESENT`, I would additionally impose temporal persistence or transition constraints. A two-second spike should not create a clinical phase segment. This can be implemented with class-specific duration priors in a semi-Markov decoder.

---

# 3. AWS Cloud Architecture & Cost Optimization

## 3.1 Proposed scale architecture

```text
┌──────────────────── Hospital ────────────────────┐
│  Camera 1 ─┐                                      │
│  Camera 2 ─┼─► Gateway: timestamp / auth / QoS   │
│  Camera 3 ─┘                                      │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
              AWS video ingestion
              (Kinesis Video Streams)
                       │
             ephemeral consumer path
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   decode / sampling          stream metadata
          │                         │
          ▼                         ▼
  GPU/CPU feature service      synchronization
          │                         │
          └────────────┬────────────┘
                       ▼
              time-aligned features
                       │
                       ▼
              inference service
       (ECS/EKS GPU or SageMaker)
                       │
          ┌────────────┼─────────────┐
          ▼            ▼             ▼
     live timeline   hard-event    observability
       output         stream       / metrics
          │            │
          ▼            ▼
   application API   Kinesis /
                    S3 feature store
                         │
                         ▼
                offline active learning
                         │
                         ▼
                  annotation queue
```

### Concrete AWS service roles

**Kinesis Video Streams**: managed ingestion of live video streams.

**Amazon ECS on EC2 GPU instances** or **Amazon EKS GPU nodes**: suitable for a continuously warm, multi-camera inference fleet where GPU utilization can be packed across many rooms.

**Amazon SageMaker real-time endpoints**: attractive when the organization wants managed model deployment, model versions, monitoring, and autoscaling. I would benchmark it against ECS for the specific low-latency and utilization profile.

**Kinesis Data Streams**: derived feature/event transport when the organization needs durable streaming fan-out.

**S3**: offline feature datasets, annotation manifests, model artifacts, and governance-approved historical data. Raw video should have explicit lifecycle/retention policies.

**DynamoDB**: low-latency session state, camera configuration, model version, and workflow metadata.

**CloudWatch**: latency, GPU utilization, dropped frames, camera synchronization quality, model confidence, and service health.

**AWS IAM + KMS**: least-privilege access and encryption.

**EventBridge / Step Functions**: orchestration for batch curation, annotation jobs, retraining workflows, and approval gates.

### Synchronization and jitter

Each camera frame should carry:

- capture timestamp;
- gateway receive timestamp;
- sequence number;
- camera ID.

The synchronizer maintains a small bounded buffer per camera. It forms a timestep when frames fall inside a configurable tolerance window. If one camera is late, the system should not block indefinitely.

Instead:

1. wait for a bounded jitter budget;
2. use the newest valid frame within tolerance;
3. mark the camera quality as degraded;
4. optionally reuse the last embedding for a very short period;
5. continue inference.

This converts network jitter into an explicit model feature rather than an invisible source of corruption.

## 3.2 Cost optimization

### 1. Separate decoding from expensive inference

Do not run the largest model on every frame from every camera.

A cheap stage can detect:

- no meaningful scene change;
- empty OR;
- stable `PATIENT_PRESENT`;
- camera failure.

Only uncertain windows go to the expensive temporal model.

### 2. Adaptive sampling

Use different sampling rates by state:

- low rate during confidently static periods;
- normal rate around workflow transitions;
- high rate around suspected `OPERATION` boundaries.

This is especially effective because `PATIENT_PRESENT` is explicitly long and static.

### 3. Pack multiple ORs per accelerator

Rather than one GPU per room, schedule independent room streams onto shared GPU workers. Micro-batching feature inference across rooms can substantially improve utilization.

The target should be **GPU utilization and latency SLOs**, not simply the number of deployed instances.

### 4. Use small models for always-on inference

A small temporal model can remain always-on. A larger model can be invoked only for:

- uncertain windows;
- boundary refinement;
- camera disagreement;
- rare workflows.

### 5. Feature caching instead of raw-video caching

Cache compact embeddings only where governance allows. This reduces network and compute cost for downstream experimentation while respecting a shorter/stricter retention policy.

### 6. Autoscaling with hysteresis

Autoscaling every few seconds creates instability. Scale based on queue depth, GPU utilization, and active-room count with cooldown periods. Maintain a small warm pool to protect latency.

### 7. Quantization / distillation

The production temporal model should be distilled into a smaller student model and evaluated under realistic boundary metrics. INT8 inference is worth considering after validating that calibration does not damage rare-class recall or boundary accuracy.

---

# 4. Evaluation Strategy

Frame accuracy alone is insufficient. A model can achieve high accuracy by correctly predicting a long static `PATIENT_PRESENT` interval while still producing unusable fragmented timelines.

I would report two metric groups.

## Model quality

- frame accuracy;
- macro F1;
- per-class F1;
- confusion matrix;
- calibration / expected calibration error.

## Product quality

- segment IoU;
- boundary MAE in seconds;
- fragmentation rate;
- false-positive seconds for `PATIENT_PRESENT`;
- missed-operation duration;
- transition detection recall within ±N seconds;
- percentage of timelines satisfying clinical workflow constraints.

For the two known failure modes, I would make **Patient Present FP duration** and **Operation boundary MAE** release-blocking metrics rather than secondary dashboards.

Evaluation should be split by hospital, room, camera configuration, procedure type, and time period to avoid leakage. A random frame split would overestimate performance because adjacent frames are highly correlated.

---

# 5. Prototype Mapping to the Repository

The prototype intentionally mirrors the production interfaces:

- `data.py`: mock multi-camera feature ingestion;
- `model.py`: quality-aware fusion and GRU temporal classifier;
- `postprocess.py`: frame-to-segment cleanup;
- `metrics.py`: model + product metrics;
- `pipeline.py`: deterministic train/evaluate orchestration;
- `scripts/error_analysis.py`: targeted failure simulation;
- `tests/`: behavior-level tests.

The implementation is intentionally small enough to review in minutes, while each interface can later be replaced independently by production components.

## Key engineering trade-offs

**Feature-level fusion over pixel fusion:** more robust to occlusion and cheaper to operate; gives up some fine-grained spatial alignment.

**GRU prototype over a large Transformer:** easier to run and explain in a seven-hour challenge; production should benchmark TCN/local-attention alternatives.

**Rule-based post-processing:** deterministic and easy to test; production should evaluate a learned/semi-Markov boundary decoder.

**Derived hard-example events over retained video:** aligns the ML lifecycle with the retention constraint; requires careful privacy review because embeddings can still be sensitive.

**Shared GPU fleet over per-room endpoints:** materially better utilization at scale; requires robust multi-tenant scheduling and isolation.

---

# 6. Recommended Next Experiments

1. Compare GRU, dilated TCN, and local-attention temporal models at equal latency/compute.
2. Add a dedicated boundary head and measure boundary MAE before/after.
3. Compare mean, quality-weighted, and attention-based camera fusion under synthetic occlusion.
4. Run active-learning simulations measuring F1/boundary improvement per labeled minute.
5. Establish production SLOs: end-to-end latency, dropped-frame rate, synchronization error, and maximum acceptable timeline fragmentation.
6. Validate the retention-safe hard-example mechanism with privacy/security stakeholders before deployment.
