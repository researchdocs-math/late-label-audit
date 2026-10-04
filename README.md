# Late Labels, Hidden Errors

Reproducible experiments for completion-aware auditing of machine-learning prediction cohorts. The implementation combines exact hypergeometric intervals with all possible outcomes of unfinished audit requests, then intersects the result with deterministic bounds from known outcomes.

## Quick start

Use Python 3.12 (the recorded environment) or a compatible Python 3.11 installation. Run the commands from this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/verify.py --replay
python src/aggregate.py
```

The archive includes processed features, frozen prediction records, and the released numerical results. The quick start runs exact-coverage tests, validates file hashes, independently repeats the first two full replay seeds of all five prediction streams, and reconstructs the primary numerical summary.

## Repeat every monitoring experiment

```sh
python src/experiment.py --seeds 200 --workers 10 --output main
python src/synthetic.py --reps 2000 --workers 10
python src/aggregate.py
python src/verify.py
```

On an Apple M3 Pro with 18 GB memory, the recorded main run took about two minutes with ten worker processes. Reduce `--workers` for smaller machines. Numerical libraries are limited to one thread per replay worker to avoid nested oversubscription. Run times depend on hardware and cache state.

The main study uses five fixed prediction streams, eight verification conditions, four budgets, seven observation times, and seven monitoring rules. There are 200 delay-and-audit seeds per stream. The synthetic study uses 48 combinations of fixed risk, backlog fraction, and delay condition, with 2,000 repetitions each. The random generators are NumPy PCG64 initialized using explicit SeedSequence components in the source. Methods share requests and delays within each replay.

The experiment-level confidence budget is 0.05, divided over the 48 gas or 12 hydraulic cohorts. Each budget is a separate fixed-budget policy. The decision limit is 0.15. The synthetic study assesses a single cohort at confidence level 0.95. The main four-stream comparison uses the two CPU models on each dataset; the gas MLP is an additional sensitivity stream.

## Regenerate predictions

To retrain the CPU models from the included processed features:

```sh
python src/prepare.py --features-only --workers 10
```

To regenerate features from the original public archives:

```sh
python src/download.py
python src/prepare.py --workers 10
```

To also retrain the neural gas classifier using Apple's Metal backend when available:

```sh
python -m pip install -r requirements-gpu.txt
python src/prepare.py --features-only --workers 10 --mlp
```

The neural model uses seed 1729, 400 full-batch Adam steps, and the architecture and optimization settings recorded in `src/prepare.py`. GPU kernels can differ numerically across PyTorch, operating-system, and hardware versions. The included frozen prediction CSVs define the exact inputs used by the monitoring experiments; the model weights are also included. Retraining can replace these CSVs, which intentionally changes their checksum if the resulting predictions differ.

## Files

- `src/bounds.py`: exact hypergeometric inversion, incomplete-panel envelope, and deterministic interval intersection.
- `src/experiment.py`: paired sensor-cohort simulation and seed-level metrics.
- `src/synthetic.py`: controlled finite-population and verification-delay simulation.
- `src/prepare.py`: source feature processing and classifier training.
- `src/download.py`: public dataset download with source-archive checksum validation.
- `src/aggregate.py`: primary effect sizes and Monte Carlo uncertainty.
- `src/verify.py`: checksum validation and independent replay comparison.
- `tests/test_bounds.py`: exact combinatorial coverage, all feasible censoring-count containment, census and zero-sample cases, and an arrival-selection counterexample.
- `data/features.npz`: gas feature vectors, hydraulic sensor summaries, and source labels/regimes.
- `data/*_lr.csv`, `data/*_rf.csv`, `data/gas_mlp.csv`: immutable prediction records with source row ID, regime, reference label, predicted class, confidence, and 0/1 error.
- `data/gas_mlp_weights.pt`: trained neural-model state and input-standardization parameters.
- `data/PROVENANCE.md`: dataset creators, source identifiers, transformations, licenses, and original archive hashes.
- `results/training.json`: model settings, training counts, errors, timing, and library versions.
- `results/main/config.json`: full main-run configuration and prediction hashes.
- `results/main/seed_metrics.csv.gz`: one row per stream, seed, condition, budget, time, and rule; cohort-averaged metrics.
- `results/main/simultaneous.csv`: any failure and any incorrect decision over all cohorts and times in each replay.
- `results/main/example_paths.csv.gz`: full cohort-level trajectories for seed zero.
- `results/main/summary.csv`: aggregate means and standard deviations.
- `results/synthetic.csv`: cell means, Monte Carlo standard errors, and all-times failure frequencies.
- `results/synthetic_config.json`: synthetic design and execution details.
- `results/numerical_summary.json`: reconstructed primary comparison.
- `checksums.json`: hashes of the fixed inputs and numerical outputs.
- `CITATION.cff`, `LICENSE`, and the requirements files: citation, MIT code license, and exact dependency versions.

## Metric interpretation

`truth` is the realized error fraction of the frozen cohort. `failure` means that the interval misses this fraction. `correct` and `wrong` describe decisions relative to the error limit; an unresolved result counts as neither. `any_failure` and `any_wrong` refer to an entire replay, across times and cohorts. `requested` counts audit queries charged to the rule and `returned` counts observed labels among those queries; routine-only and no-audit rules have zero audit charges. `routine_rate` describes the selected routine responses and is recorded as a diagnostic.

Stream summaries weight cohorts equally. The main effect weights the four primary streams equally and estimates simulation uncertainty over paired seeds. These intervals concern random verification schedules and audits conditional on the data and models. They do not represent uncertainty over unseen institutions, sensor platforms, or future operating conditions.

## Statistical contract

The initial pending frame and panel budget are fixed before audit outcomes are observed. The panel is uniform without replacement. Reference outcomes are correct and stable. Delays may depend on hidden errors and may be infinite. The method certifies the realized risk of the frozen cohort. Incomplete information can appropriately leave the result unresolved.

## License and citation

Copyright 2026 Cemil Emre Yavas. The implementation is released under the MIT license. Source-derived data are credited separately in `data/PROVENANCE.md`. Use `CITATION.cff` to cite this release. Contact: cemil@fairesresearch.org.
