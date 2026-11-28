# Rate monitor: is activation-based error detection stronger for the SNN than the ANN?

The temporal-spike pilot found one signal: adding internal firing statistics to classifier
confidence improved held-out error detection for the SNN, from AUROC 0.775 to 0.811. This experiment asks:

1. **Primary:** is that gain larger for the SNN than for the equivalent ANN monitor, or is it
   a generic activation-based effect? Generic activation monitors are known: Chen et al.,
   AISTATS 2019, use whitebox meta-models on intermediate-layer probes.
2. **Secondary:** does a sparse monitor (few channels) keep the gain, and at what measured cost?

CPU only. It reads the geometry experiment's saved features read-only, and trains no network.

## What the features are

| | ANN | SNN |
|---|---|---|
| `layer1-3`, `penultimate` | spatially averaged ReLU stage outputs | SEW stage outputs (sums of spike trains, 0..3) averaged over **space and time** |
| `*_spikes` | — | the same, averaged over space **at each timestep** (not binary spikes) |

`run_monitor.py` verifies these before anything else:
- every required file exists, with the right shapes
- image order and labels are identical across models and conditions
- `preds = argmax(logits)`
- the SNN logits equal fc(penultimate)
- the SNN per-timestep values average to the time-averaged ones
- the fit and eval images are disjoint

The result is written to `feature_check.json`.

## Design

| Set | Features | Models |
|---|---|---|
| A | MSP, entropy, top-2 margin | both |
| B | A + 960 pooled channel values (log1p) | both |
| C | B + per-layer Mahalanobis distance to the nearest clean class mean (fit on clean training images) | both |
| D | C + per-timestep channel values + confidence trajectory | SNN only |

- **Protocol:** leave one corruption out, with disjoint fit/eval images. The following are fit on training rows only:
  - standardisation, C selection (grouped CV), coefficients
  - margin-decile edges
  - channel rankings

  A test scrambles every evaluation row and checks that none of these change.
- **Main statistic:** [AUROC_SNN(B) − AUROC_SNN(A)] − [AUROC_ANN(B) − AUROC_ANN(A)]. It is computed on the same images, with each model scored against **its own** errors. It is reported per cell, per held-out corruption, and pooled. The bootstrap is paired: each cell's resamples are drawn once and shared by every model, set and statistic.
- **Sparse monitors:** at 10, 19, 48, 96 and 240 channels, three ways of choosing channels are compared, all on training rows:
  - order of entry on an L1 path
  - a univariate AUROC filter (supervised baseline)
  - 5 random subsets
- **Cost:** `benchmark_overhead.py` times the real networks on CPU. The monitor step is timed in isolation on captured layer outputs, because its cost is far below the jitter of a forward pass. The timings are reported as % of a forward pass, along with feature and parameter bytes.
- **Screening rules** (exploratory, written before the run, in `decision.json`):
  - "SNN-specific": the B−A difference CI is above 0 in ≥ 3 of 6 held-out corruptions, or D−C ≥ 0.01 margin-stratified in ≥ 2 of 6
  - "sparse": 10% of channels keep ≥ 90% of the gain and beat random subsets

  These involve many comparisons without correction and are not significance tests. A positive SNN−ANN difference would not by itself show a spiking-specific mechanism.

## Running

Local smoke test on synthetic features. These have the exact on-disk layout, with an error signal planted in one SNN channel:

```
python experiments/rate_monitor/make_synthetic.py experiments/rate_monitor/synthetic_features
cd experiments/rate_monitor
python run_monitor.py --stage smoke --features-root synthetic_features
python benchmark_overhead.py --stage smoke --random-init
python make_figures.py --stage smoke
python -m pytest tests -q          # 15 tests
```

### Kaggle (CPU notebook)

1. Create a new notebook. Settings: Accelerator **None** (CPU), Internet **on**.
2. Add Input: **notebookc5a36e39e3**, the version whose output has `snn-geometry/features/` and
   `snn-geometry/checkpoints/`.
3. Cell 1:
   ```python
   !git clone https://github.com/bishalpanthi321/snn-geometry /kaggle/working/snn-geometry
   %cd /kaggle/working/snn-geometry
   !pip install -q "spikingjelly @ git+https://github.com/fangwei123456/spikingjelly.git@ea10208738a48d905f78589040593d886bacb067"
   import glob, os
   feat = glob.glob('/kaggle/input/*/*/*/snn-geometry/features') or glob.glob('/kaggle/input/*/snn-geometry/features')
   ck = glob.glob('/kaggle/input/*/*/*/snn-geometry/checkpoints') or glob.glob('/kaggle/input/*/snn-geometry/checkpoints')
   print(feat, ck, sep='\n')
   assert feat and all(os.path.isdir(os.path.join(feat[0], t, 'eval')) for t in ('ann', 'snn')), \
       "features not found: attach the notebookc5a36e39e3 version that has features/"
   os.environ['RM_FEATURES'] = feat[0]
   if ck:
       os.environ['RM_CHECKPOINT_DIR'] = ck[0]
   os.environ['RM_OUT_ROOT'] = '/kaggle/working/rm_out'
   ```
4. Cell 2:
   ```python
   !bash experiments/rate_monitor/run_pilot.sh pilot 2>&1 | tee /kaggle/working/rate_monitor.log
   ```
5. Save Version → **Save & Run All (Commit)**. Results go to `/kaggle/working/rm_out/results/pilot/`.

The pilot uses 2000 fit and 2000 eval images per condition, at severities 1, 3 and 5. Estimated time is 45–90 min
on Kaggle's CPU. This is unmeasured: the real time is written to `decision.json` and to the log.
