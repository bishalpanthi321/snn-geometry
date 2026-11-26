# Corruption geometry in ANN vs SNN representations (CIFAR-10)

Tests whether input corruptions move hidden representations along a few directions
that are roughly orthogonal to the class-discriminative (LDA) subspace, whether
projecting those directions out recovers accuracy, and whether a spiking network's
time dimension changes the picture. Hypotheses H1–H4 and verdict rules are in
`src/summary.py`; results land in `results/summary.md`.

## Pipeline

```
config.yaml -> train.py (--model ann|snn) -> generate_corruptions.py -> extract_features.py -> analyze.py -> make_figures.py
```

Every stage reads its inputs from disk and writes its outputs to disk, so any stage can be rerun alone.
`--smoke` on any script runs the toy configuration and writes under `smoke_run/` instead.

| Stage | Command | Output | Time on RTX 3050 Laptop (4 GB) |
|---|---|---|---|
| tests | `python -m pytest -q` | — | ~45 s |
| train ANN | `python train.py --model ann` | `checkpoints/ann/{last,best,final}.pt`, `logs/train_ann.csv` | measured 1,560 img/s → ~32 s/epoch → **~55 min** for 100 epochs (extrapolated from smoke test) |
| train SNN | `python train.py --model snn --micro-batch 32` | same under `snn/` | measured 143 img/s at micro-batch 32 → ~350 s/epoch → **~10 h** (extrapolated) |
| corruptions | `python generate_corruptions.py [--cifar10c]` | `data/corrupted/{corruption}/{severity}.npy` | ~5 min for 30 sets (smoke: 4 sets in 38 s) |
| features | `python extract_features.py --model ann` / `--model snn` / `--model snn --T 8` | `features/{ann,snn,snn_T8}/{train,fit,eval}/...` | smoke: 25 s / 30 s / 56 s; full run not yet timed |
| analysis (CPU) | `python analyze.py` | `results/*.csv`, `results/summary.md` | smoke: 78 s; full run not yet timed |
| figures | `python make_figures.py` | `results/figures/*.png` | < 1 min |

Full-run times will be filled in from the real logs after the run.

Useful flags: `--resume` (continue from `last.pt`), `--max-hours H` (stop cleanly after the epoch
that crosses H hours; rerun with `--resume`), `--micro-batch M` (gradient accumulation for small GPUs),
`--out-root DIR` (or env `SNNGEO_OUT_ROOT`) to write checkpoints/features/results elsewhere.

`run_smoke.ps1` runs the whole toy pipeline on Windows; `run_all.sh` runs the full pipeline on Linux
and resumes training if a checkpoint exists.

## Setup

```
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
```

## Running on a free cloud GPU (Kaggle)

Kaggle notebooks give a free T4 (16 GB) or 2xT4, sessions of up to 12 h, and roughly 30 GPU-hours per
week (the exact quota is shown in your account settings). GPU and internet access need a
phone-verified account. On a T4 the SNN fits the full batch of 128 (measured peak ≈ 34 MB per image
→ ≈ 4.3 GB), so no gradient accumulation is needed and both models train with identical BatchNorm
batch size.

1. Put this folder on GitHub (private is fine) or upload it as a Kaggle Dataset (zip).
2. New notebook → Settings: Accelerator = GPU T4 x2, Internet = on.
3. Cell:
   ```
   !git clone <your repo> /kaggle/working/snn-geometry   # or copy from /kaggle/input/...
   %cd /kaggle/working/snn-geometry
   !pip install -q "spikingjelly @ git+https://github.com/fangwei123456/spikingjelly.git@ea10208738a48d905f78589040593d886bacb067"
   !python -m pytest -q
   # two GPUs: train both models at the same time
   !CUDA_VISIBLE_DEVICES=0 nohup python train.py --model ann --max-hours 11.3 > ann.out 2>&1 &
   !CUDA_VISIBLE_DEVICES=1 python train.py --model snn --max-hours 11.3
   ```
4. Use "Save Version → Save & Run All (commit)" so it runs with the browser closed. Outputs in
   `/kaggle/working` are kept with the version.
5. If the SNN did not finish within one session, start a new notebook with the previous version's
   output attached as input, copy `checkpoints/` back into the working dir and run
   `python train.py --model snn --resume --max-hours 11.3`.
6. Then run extraction + analysis there (or download `checkpoints/*/final.pt` and run them locally).
   Full features take roughly 14 GB; Kaggle's `/kaggle/working` limit is 20 GB.

## Design decisions worth knowing

- **ANN**: standard CIFAR ResNet-18 (ReLU after the residual sum). **SNN**: SEW-ResNet-18 with ADD,
  LIF on each branch, BN after every conv. These are the standard definitions of the two models; the
  only structural difference beyond the neuron is where the nonlinearity sits relative to the sum,
  which SEW requires.
- **LIF**: spikingjelly 2.0.0 `LIFNode`, tau=2 (decay 0.5), `decay_input=False` (so v = 0.5 v + x),
  hard reset to 0, threshold 1, ATan surrogate, detached reset. A plain-PyTorch LIF in
  `src/neurons.py` is tested to match spikingjelly exactly (forward, membrane, gradients) and is
  used automatically if spikingjelly cannot be imported.
- **State reset**: the SNN resets every neuron at the start of each `forward`, so state cannot leak
  between batches (tested, including a control showing that without the reset it *does* leak).
- **Direct encoding**: stem conv+BN computed once and repeated T times (identical to repeating the
  image; cheaper).
- **Logits** = mean over t of fc(pooled spikes at t) = fc(time-averaged rate). Verified in a unit
  test and on every extracted batch (max error ≈ 2e-6 in fp32).
- **layer4 vs penultimate**: the pooled layer4 output *is* the penultimate feature, so it is stored
  once as `penultimate` (an exact-equality assert runs on every batch).
- **Corruptions**: CIFAR-10-C constants from Hendrycks' `make_cifar_c.py`; results are rounded to
  uint8 instead of truncated (truncation adds a −0.5 level shift to every pixel). `--cifar10c`
  downloads the official set and writes a pixel-level cross-check to `results/cifar10c_crosscheck.csv`.
  Result of that check (run 2026-11-25): for contrast, gaussian_blur and brightness no pixel differs
  from the official files by more than 1 level (mean difference 0.50 = rounding vs truncation); for
  the three noise types the RMS change from clean matches the official set within 0.3% at every
  severity (pixels differ because the random draws differ); official labels match test-set order.
- **Splits**: test set split 50/50 (fixed seed) into fit/eval; all subspaces are fit on fit, every
  metric is reported on eval. LDA is fit on clean training features. For H3 the number of removed
  directions k is chosen on the fit split, then reported on eval.
- **Checkpoint for analysis**: `final.pt` by default (`best.pt` is selected on the test set).
- **Determinism**: seeds are fixed and logged; training uses `cudnn.benchmark` so two training runs
  are not bit-identical. Extraction is deterministic (repeat check: max difference 0).
