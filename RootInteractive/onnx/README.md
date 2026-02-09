# Phase 0.1.E — ONNX Cross-Backend Invariance & Performance Framework

Three-way cross-backend invariance tests for ONNX model inference:
**sklearn** ↔ **Python onnxruntime** ↔ **Node.js onnxruntime-node**

## Strategic Context

See `RootInteractive_ONNX_Client_Side_Vision_v1_0.md` for motivation:
client-side ONNX inference enables interactive systematic distortion analysis
on O(10⁴–10⁷) data points for both real data and Monte Carlo.

## Setup

### Python dependencies

```bash
pip install onnxruntime skl2onnx scikit-learn --break-system-packages
```

### Node.js dependencies

```bash
cd RootInteractive/RootInteractive/onnx/
npm install
```

### Generate reference data (run once)

```bash
python generate_models.py
```

This produces 7 `.onnx` models and reference predictions in `reference_data/`.
**Models are cached artifacts — tests never retrain.**

## Running Tests

### Correctness tests only (CI, < 60s)

```bash
pytest test_invariance_onnx.py -m "not bench" -v
```

### Full test suite including benchmarks

```bash
pytest test_invariance_onnx.py -v
```

### Benchmarks standalone (Node.js)

```bash
node bench_inference_onnx.mjs                    # All models
node bench_inference_onnx.mjs --model ridge      # Single model
```

Results saved to `bench_results/bench_onnx_results.json`.

## Models

| Model | Task | sklearn Class | Expected .onnx Size |
|-------|------|---------------|---------------------|
| Ridge(α=0.1) | Regression | Ridge | ~300 B |
| RF(10, d=3) | Regression | RandomForestRegressor | ~6 KB |
| RF(50, d=5) | Regression | RandomForestRegressor | ~116 KB |
| MLP(100,50) | Regression | MLPRegressor | ~23 KB |
| LogisticRegression | Classification | LogisticRegression | ~400 B |
| RF Clf(10, d=3) | Classification | RandomForestClassifier | ~6 KB |
| MLP Clf(50,25) | Classification | MLPClassifier | ~8 KB |

## Performance Budgets

| Tier | Budget | Use Case |
|------|--------|----------|
| Interactive | < 75 ms/call | Slider-driven distortion exploration |
| Batch | < 1 s/call | Reference dataset computation |

## File Structure

```
onnx/
├── models/                      # Generated .onnx files (cached)
├── reference_data/              # Test vectors + precision budget (cached)
├── bench_results/               # Benchmark output
├── generate_models.py           # Train → export → reference data
├── test_invariance_onnx.py      # pytest: 3-way invariance tests
├── test_cross_backend_onnx.mjs  # Node.js ONNX runner
├── bench_inference_onnx.mjs     # Performance benchmarks
├── conftest.py                  # pytest configuration
├── package.json                 # npm dependencies
└── README.md                    # This file
```

## Version Requirements

- Python ≥ 3.9
- Node.js ≥ 18
- onnxruntime (Python) ≥ 1.17
- onnxruntime-node ≥ 1.17
- skl2onnx ≥ 1.16
- ONNX opset: 17
