# Regime Pipeline

This repository bundles two tightly-coupled stages into a single workflow:

- **Part 1 – Regime Detection (`regime_pipeline.regime_detection`)**  
  Fit a Markov-switching model on benchmark returns to infer risk-on/off states and produce a tradable regime signal.

- **Part 2 – Sector Rotation (`regime_pipeline.sector_rotation`)**  
  Combine the regime signal with momentum-driven sector allocation to build a complete portfolio backtest.

Running both parts sequentially delivers a reproducible pipeline from raw prices to portfolio-level analytics.

---

## Project Layout

```
regime_pipeline/
  regime_detection/   # Part 1 modules (model fitting, plots, CLI helper)
  sector_rotation/    # Part 2 modules (data, signals, allocators, reporting)
scripts/
  run_regime_detection.py   # Execute Part 1 end-to-end
  run_sector_rotation.py    # Execute Part 2 using the Part 1 signal
configs/
  regime_detection.yaml     # Default parameters for Part 1
  sector_rotation.yaml      # Default parameters for Part 2
requirements/
  regime_detection.txt
  sector_rotation.txt
```

All shared utilities live under the `regime_pipeline` package, so Part 2 directly reuses the Part 1 regime output instead of reimplementing a classifier.

---

## Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run Part 1 – Regime detection**

   ```bash
   python scripts/run_regime_detection.py \
     --config configs/regime_detection.yaml \
     --output-dir artifacts/regime_detection
   ```

   This stage downloads benchmark data, fits the Markov-switching model, and stores probabilities, a binary risk signal, and summary statistics under `artifacts/regime_detection/`.

3. **Run Part 2 – Sector rotation**

   ```bash
   python scripts/run_sector_rotation.py \
     --config configs/sector_rotation.yaml \
     --regime-artifacts artifacts/regime_detection
   ```

   The sector-rotation run reuses (or regenerates) the cached regime signal and produces portfolio weights, an equity curve, and a tear sheet in `data/`.

---

## Customisation Tips

- **Modify configurations** – Both stages load YAML configs from the `configs/` directory. Copy these files and pass alternative paths via `--config` to test new universes, thresholds, and risk controls.
- **Segmented requirements** – Use `requirements/regime_detection.txt` or `requirements/sector_rotation.txt` if you only need one part of the pipeline.
- **Testing** – Run `pytest` to validate the sector-rotation utilities after making changes.

---

## Next Steps

- Add additional macro features to `regime_detection` and feed them into allocation decisions.
- Extend the pipeline to run scenario sweeps and capture results in a reporting notebook.
- Wire the two parts into CI (download caching recommended) to guard future refactors.