# ACO-CNN experiment analysis

Evaluator type: `cnn_mnist`
Trial count: `600`
Best validation accuracy: `0.9860000014305115`
Per-run summaries: `6`
Failed trials: `0`
Cache hits: `333`
Cache hit rate: `0.549`
Test evaluations recorded by search report: `0`

## Methodological notes

- Journal facts: MNIST, 20 ants, pheromone-only selection, rho=0.25, constant +0.5 reinforcement, and the eight-dimensional paper search space.
- Interpretations: paper_literal updates after every ant; paper_conventional updates once per iteration. Neither is evidence of the authors' unavailable original code.
- Development decisions: fixed CNN architecture, deterministic 50,000/10,000 validation split, validation fitness, caching, early stopping, repeated seeds, and untouched final test evaluation.
- Smoke and pilot runs validate implementation behavior and are not primary scientific results.
- paper_conventional versus improved is a pipeline comparison, not a controlled pheromone-update ablation, because their search spaces differ.
- Budget provenance: 20 ants follows the explicit journal setting; the selected iteration count and maximum epoch count are project computational-budget decisions because the article does not specify them.
- The reduced budget targets a feasible reproducible three-seed experiment on available hardware; it does not claim literal replication of the article's runtime.
- paper_conventional versus improved is a pipeline comparison: their search spaces differ and the result is not a controlled pheromone-update ablation.

## Final evaluation

- Final evaluation status: `success`
- Final test evaluations: `6`
- `paper_conventional/seed_42` test accuracy: `0.9855999946594238`
- `paper_conventional/seed_43` test accuracy: `0.983299970626831`
- `paper_conventional/seed_44` test accuracy: `0.9866999983787537`
- `improved/seed_42` test accuracy: `0.9627000093460083`
- `improved/seed_43` test accuracy: `0.9764000177383423`
- `improved/seed_44` test accuracy: `0.9761000275611877`

## Environment and protocol

- Evaluator type: `cnn_mnist`. Runs remain grouped by budget-aware `run_id` in the summary and source logs.
- Search runtime across mode/seed runs: `5298.029338993998` seconds.
- Primary workflow runtime (tuning plus confirmation): `5628.452621121993` seconds; candidate evaluations: `600`; confirmation evaluations: `18`.
- gpu_available: `true`
- gpu_devices: `/physical_device:GPU:0`
- gpu_utilization: `not_sampled`
- keras_version: `3.15.1`
- numpy_version: `2.5.3`
- platform: `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43`
- python_version: `3.13.15`
- tensorflow_version: `2.21.0`
- Confirmation failed trials: `0`; confirmation cache hits: `6`.
- Confirmation runtime: `330.2413777949987` seconds.
- `paper_conventional` budget: `main`; effective `20 ants × 5 iterations × 5 epochs`.
- `paper_conventional/seed_42` split `unknown`.
- `paper_conventional/seed_43` split `unknown`.
- `paper_conventional/seed_44` split `mnist-stratified-8000-2000-seed-2024`.
- `improved` budget: `main`; effective `20 ants × 5 iterations × 5 epochs`.
- `improved/seed_42` split `unknown`.
- `improved/seed_43` split `mnist-stratified-8000-2000-seed-2024`.
- `improved/seed_44` split `mnist-stratified-8000-2000-seed-2024`.
- Final policy: retrain on 60,000 development images for the confirmation best epoch, then evaluate once on the untouched 10,000-image test set.
- Dataset protocol: `MNIST`, seed `2024`, split `8000/2000/10000` (train/validation/test), split IDs `['mnist-stratified-8000-2000-seed-2024']`.
- Final evaluation status: `success`; runtime: `143.16641008499573` seconds; failed seeds: `0`.
- The untouched test set is an evaluation-only dataset and must not be used to select configurations.
