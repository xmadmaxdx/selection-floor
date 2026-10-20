# The Selection Floor

Code and manuscript for a study of **model selection in small-data deep learning**.

Every paper in the small-data regime picks a checkpoint — and usually a loss, an
architecture, an epoch count — using a validation set of a few thousand examples,
then reports the winner's test number as though it were clean. It is not clean.
The reported number is the maximum of many noisy validation estimates, so it
carries an upward bias that the test-set confidence interval does not measure.

This project derives that bias as a **closed form**, measures it **exactly** by
replaying the selection step thousands of times without retraining, and removes
most of it with a **selection-aware protocol**.

Distinct from the companion project `medmnist-c-cal` (Project 1), which measured
calibration under corruption on medical images. No shared dataset, loss, or claim.

## The floor

Two checkpoints of the same accuracy compared on $n_{val}$ examples have a gap
with standard error $SE = \sqrt{2p(1-p)/n_{val}}$. Two estimands follow from it,
and they are not interchangeable:

$$\underbrace{z_{power} \cdot SE}_{\text{ranking gap: rank correctly with prob. } 0.95}
\qquad
\underbrace{(z_{1-\alpha/2} + z_{power}) \cdot SE}_{\text{MDE: conventional, } \alpha = 0.05}$$

At $p = 0.95$ and $n_{val} = 1{,}000$ the MDE is **3.51 percentage points**. At
$p = 0.765$ and $n_{val} = 1{,}003$ it is **6.83 points**. Running more seeds
does not move either number, because the noise belongs to the validation
*sample*, not the training run.

This is a **pairwise** quantity. The protocol practitioners actually use is an
argmax over ~20 epochs, which needs a larger gap — so treat it as a lower bound
on the real floor.

Print the table yourself rather than trusting any transcription:

```bash
python -m scripts.dump_tables --mdi-table
```

## Replay

One training run records a boolean matrix of shape
`n_epochs × n_pool` — whether the model was right on each selection-pool example
at each epoch. That matrix contains the entire accuracy-based, single-run
selection experiment. Sampling validation subsets from it, selecting a checkpoint
using each, and scoring on the untouched test set gives the exact sampling
distribution of the reported number.

The measured quantity is **optimism**: the validation accuracy of the selected
checkpoint minus its test accuracy. Optimism is positive by construction,
because the checkpoint was chosen for scoring well on that draw. `regret` — how
far the pick landed from the best available — is reported separately, because it
is non-positive by construction and answers a different question.

## Protocol

| Rule | Definition |
|---|---|
| `argmax` | earliest epoch with best validation accuracy (what most work does) |
| `one_se` | earliest epoch within one standard error of the best (Hastie et al. 2001) |
| `dense_avg` | weight-space average of the top-$q$ epochs by validation loss |

The exact McNemar test is used to decide what may be *reported*, not what to
*select*. It is deliberately not a selection rule: an earlier draft included one,
and it was provably identical to `argmax`. See §8 of `GapAnalysis.md`.

## Layout

- `src/data/` — fit / pool / test split design, where the pool is never trained on
- `src/models/` — SimpleCNN and ResNet-18-small
- `src/losses/` — the candidate set a practitioner selects among
- `src/engine/trainer.py` — records the per-epoch correctness matrices, validation loss, timing, and provenance
- `src/engine/selector.py` — selection rules, replay, and the paired rule comparison
- `src/engine/averaging.py` — dense weight averaging of the top-$q$ epochs by validation loss
- `src/engine/trainer.py::floor_stop_epoch` — the compute-saving stopping rule
- `src/stats/inference.py` — MDE, McNemar, paired standard error, floor tables
- `scripts/` — train, selection_study, evaluate, run_matrix, download_data, dump_tables, regen_plots
- `tests/` — offline tests, no downloads, no GPU
- `paper/` — manuscript

## Local smoke test

```bash
pip install -r requirements.txt
python -m pytest tests/test_smoke.py -q
python -m scripts.train --config configs/smoke.yaml
python -m scripts.selection_study --checkpoint-dir outputs/checkpoints-smoke \
    --out outputs/studies/smoke.json --trials 400 --gap-draws 80 --sizes 16,32,48
python -m scripts.evaluate --config configs/smoke.yaml --out outputs/evals/smoke.json
python -m scripts.dump_tables --glob "outputs/studies/smoke.json" --floor
```

`--keep-checkpoints` is required by `evaluate.py`, since the selected epoch is
decided after training. `configs/smoke.yaml` and `configs/mnist.yaml` set it.

## Colab

Every step below is one cell. Each cell starts with `!cd selection-floor`, because
a `!cd` inside a shell cell changes the directory only for that cell — the
notebook's working directory is untouched, so a command that omits it runs from
`/content` and cannot find `scripts` or `tests`.

```bash
!git clone https://github.com/xmadmaxdx/selection-floor.git && cd selection-floor && bash colab_setup.sh
```

Cell 1 — Drive mount (optional; the code no-ops without it):

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
```

Cell 2 — tests:

```bash
!cd selection-floor && python -m pytest tests/test_smoke.py -q
```

Cell 3 — download:

```bash
!cd selection-floor && python -m scripts.download_data --root data
```

Cell 4 — training matrix (add `--floor-stop --ckpt-tag fs` for the FloorStop arm):

```bash
!cd selection-floor && python -m scripts.run_matrix --datasets mnist,fashionmnist --losses ce,label_smoothing,focal --models simplecnn --epochs 20 --seeds 0,1,2 --keep-checkpoints --download
```

Cell 5 — replay for every trained cell:

```bash
!cd selection-floor && (ls -d outputs/ckpt-* >/dev/null 2>&1 && for d in outputs/ckpt-*; do n=$(basename "$d" | sed 's/ckpt-//'); python -m scripts.selection_study --checkpoint-dir "$d" --out "outputs/studies/$n.json"; done) || echo "NO CHECKPOINTS FOUND"
```

Cell 6 — per-cell metrics (needs `--keep-checkpoints`):

```bash
!cd selection-floor && (ls outputs/tmp-configs/*.yaml >/dev/null 2>&1 && for c in outputs/tmp-configs/*.yaml; do n=$(basename "$c" .yaml); python -m scripts.evaluate --config "$c" --download --out "outputs/evals/$n.json"; done) || echo "NO CONFIGS FOUND"
```

Cell 7 — tables and figures:

```bash
!cd selection-floor && python -m scripts.dump_tables --mdi-table
```
```bash
!cd selection-floor && python -m scripts.dump_tables --glob "outputs/studies/*.json" --floor
```
```bash
!cd selection-floor && python -m scripts.dump_tables --glob "outputs/studies/*.json"
```
```bash
!cd selection-floor && cat outputs/matrix-manifest.json
```

Cell 8 — selection across runs (H2, H4, table T5):

```bash
!cd selection-floor && python -m scripts.seed_study --out outputs/seed-studies/seed.json
```

Cell 9 — per-cell metric tables (T4) and the FloorStop cost table (T6):

```bash
!cd selection-floor && python -m scripts.dump_eval_tables --glob "outputs/evals/*.json" --floor-stop-glob "outputs/evals/fs-*.json" --out outputs/tables/evals.json
```

Cell 10 — validation curves with the rule picks marked (F4):

```bash
!cd selection-floor && python -m scripts.plot_validation_curves --ckpt-glob "outputs/ckpt-*" --outdir outputs/curves
```

Training is checkpointed per epoch and every completed cell is skipped on rerun,
so a Colab disconnect costs at most one epoch. A failed cell is recorded in
`outputs/matrix-manifest.json` and the matrix exits non-zero, so a short run can
never be mistaken for a complete one.

## Data

MNIST, FashionMNIST, Kuzushiji-MNIST and CIFAR-10 are downloaded by
`scripts/download_data.py` and are not redistributed here.