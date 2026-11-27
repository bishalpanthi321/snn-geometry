# Temporal spike structure and robustness (SEW-ResNet-18, CIFAR-10)

Does the *order* of spikes, given each neuron's spike count, carry information about
classification or about errors under corruption, beyond firing rates and confidence?
This folder is self-contained. It reuses the trained SNN checkpoint and the model, neuron
and data code in `src/` read-only, and never trains, overwrites or writes into the original
experiment.

| Part | Question | Script |
|---|---|---|
| Capacity | How much timing information is even possible at each site? | `extract.py`, `analyze_capacity.py` |
| A | Does changing spike timing (counts fixed) change predictions beyond controls? | `run_interventions.py`, `analyze_interventions.py` |
| B | Do timing features predict errors on held-out corruptions beyond confidence + rates? | `extract.py`, `analyze_detection.py` |

`make_figures.py` draws all figures. Results go to `results/<stage>/`, cached tensors to
`cache/<stage>/`, logs to `logs/<stage>/`, where stage is `smoke`, `pilot` or `full`.

## Facts about this network that shape the design

- `logits = mean_t fc(pool(layer4_out_t))` is linear and symmetric in t. Any count-preserving
  change at `layer4_out` or at `layer4.1.sn2` therefore cannot change the logits. These two are
  **null sites**, checked on every run. Experiment A intervenes upstream of at least one LIF:
  `layer3_in`, `layer4_in` (integer stage outputs) and `layer4.0.sn1` (binary).
- SEW stage outputs are sums of spike trains, with values 0..3, so they are not binary. Binary
  values exist only at LIF outputs. Every result names which kind of site it comes from.
- Direct encoding feeds the same current at every t, so stem neurons fire in a fixed order that
  depends only on input strength. The capacity analysis measures how much ordering freedom
  is left at each site.

## Interventions (Experiment A)

| Family | Keeps | Destroys |
|---|---|---|
| identity | everything (must reproduce logits bit-exactly) | none |
| same_count_swap | every neuron's count; each channel's count at every t | which position carries which order; alignment across channels |
| shared_perm (all 23 at T=4) | co-activation | temporal order |
| clean_resample | count; per-channel order distribution from clean training images | **correlations between neurons** (drawn independently) and the image's own timing |
| uniform_perm, circular_shift (wraps around), even_spread | count | timing; produces trains the network never saw |
| count_change (positive control) | — | counts, with the same L1 size as uniform_perm |

Count preservation is asserted exactly on every batch. 5 seeds per stochastic family.
Bootstrap CIs are over images. Seed spread is reported separately (`*_sd_runs`). With a
single checkpoint, no claim generalizes across training seeds or architectures.

## Detection (Experiment B)

Nested logistic models: M0 = confidence plus the confidence trajectory over t; M1 = M0 plus
firing rates and an activation Mahalanobis score; M2 = M1 plus timing given count. The test is
whether AUROC(M2) − AUROC(M1) is above zero:

- within every corruption × severity cell
- also stratified by margin and by firing count
- under leave-one-corruption-out, with disjoint fit and eval images

There is also a clean-only anomaly score fit on clean training images, without error labels.

## Pilot decision rules (fixed before running)

- **A goes to full scale** if, at an upstream site, the clean flip rate under clean_resample or
  uniform_perm has a 95% CI excluding 0 *and* differs from same_count_swap, **or** a
  margin-matched corrupted − clean flip difference has a CI excluding 0. This is a screening
  rule: it covers many comparisons and is not a significance test.
- **B goes to full scale** if the mean over severities of the margin-stratified M2 − M1 is at
  least 0.01 for at least 2 of the 6 held-out corruptions.

The verdicts are in `results/pilot/pilot_decision_{A,B}.json`.

## Running

Locally (random weights, no checkpoint needed; about 2 min plus tests):

```
powershell -File experiments/temporal_spikes/run_smoke.ps1
python -m pytest experiments/temporal_spikes/tests -q          # 95 tests, ~2 min on CPU
$env:TS_RUN_SLOW=1; python -m pytest experiments/temporal_spikes/tests/test_smoke_end_to_end.py
```

Paths: `--checkpoint/--data-root/--corrupted-dir/--out-root` flags, or the env vars
`TS_CHECKPOINT`, `TS_DATA_ROOT`, `TS_CORRUPTED_DIR`, `TS_OUT_ROOT`. The checkpoint defaults to
`checkpoints/snn/final.pt`. `extract.py` re-measures clean accuracy on all 10k test images and
stops if it differs from the original run's value (0.9168 at T=4, 0.9195 at T=8) by more than
0.003.

Every GPU stage resumes: finished conditions are skipped, and files are written atomically.

### Kaggle pilot

1. Create a new notebook. Settings: Accelerator **GPU T4 x2**, Internet **on**.
2. Under **Add Input**, choose your notebook, then the Version 1 output (it holds
   `checkpoints/snn/final.pt`). If Version 2's output has `data/corrupted/`, add that too.
3. Cell 1, setup and locating inputs:
   ```python
   !git clone https://github.com/bishalpanthi321/snn-geometry /kaggle/working/snn-geometry
   %cd /kaggle/working/snn-geometry
   !pip install -q "spikingjelly @ git+https://github.com/fangwei123456/spikingjelly.git@ea10208738a48d905f78589040593d886bacb067"
   import glob, os
   ck = glob.glob('/kaggle/input/**/checkpoints/snn/final.pt', recursive=True)
   cor = glob.glob('/kaggle/input/**/data/corrupted', recursive=True)
   cif = glob.glob('/kaggle/input/**/cifar-10-batches-py', recursive=True)
   print(ck, cor, cif, sep='\n')
   assert ck, "checkpoint not found: attach the Version 1 output"
   os.environ['TS_CHECKPOINT'] = ck[0]
   os.environ['TS_OUT_ROOT'] = '/kaggle/working/ts_out'
   os.environ['TS_DATA_ROOT'] = os.path.dirname(cif[0]) if cif else '/kaggle/working/data'
   os.environ['SNNGEO_DATA_ROOT'] = os.environ['TS_DATA_ROOT']
   if cor:
       os.environ['TS_CORRUPTED_DIR'] = cor[0]
   else:   # regenerate with the original generator and its fixed seed (identical files)
       import subprocess
       subprocess.run(['python', 'generate_corruptions.py', '--out-root', '/kaggle/working/gen'], check=True)
       os.environ['TS_CORRUPTED_DIR'] = '/kaggle/working/gen/data/corrupted'
   ```
4. Cell 2, the pilot:
   ```python
   !bash experiments/temporal_spikes/run_pilot.sh pilot 2>&1 | tee /kaggle/working/pilot.log
   ```
5. Use **Save Version → Save & Run All (commit)** so the run continues with the browser
   closed. Results are in `/kaggle/working/ts_out/results/pilot/`.

Estimated pilot cost on 2×T4: about 45–60 min. These are extrapolated estimates. The real
timings are logged in `cache/pilot/T*/meta.json` and `cache/pilot/T4/interv/meta.json`.
Output about 2.2 GB, of which the T=8 integer-site reference tables are 0.2 GB.
