# Four-Channel CNN-Micro Direct Training Summary

## Selected Configuration

- Model: `CNN_Micro`
- Input active channels: 4
- Muscle order: `[ECRB, ECRL, FDS, FDP]`
- Selected hardware channel policy: right-arm mapping
- Right-arm one-based channels: `[15, 16, 9, 1]`
- Right-arm zero-based channels: `[14, 15, 8, 0]`
- Ground/reference channel included: no
- Feature tensor contract: `(batch, 4, windows, 12)`
- CNN input contract: `(batch, windows, 48)`
- Selected context: 9 windows, corresponding to the 1-second context setting
- Output order: `[thumb, index, middle, ring, little]`

## Dataset

- Source dataset: PhysioMio impaired-arm data
- Sampling rate: 2048 Hz
- Four-channel processed tensor shape per sample before context slicing: `(4, 38, 12)`
- Train split: 2656 samples, 34 patients
- Validation split: 432 samples, 5 patients
- Test split: 720 samples, 9 patients
- Patient split leakage check: enforced by preprocessing manifests

## Selection Runs

Mapping selection used validation macro-F1 with 500 ms context:

| View | Best validation macro-F1 | Test subset accuracy | Test finger accuracy | Test macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Left | 0.6301 | 0.5054 | 0.7415 | 0.6234 |
| Right | 0.6422 | 0.4804 | 0.7438 | 0.5786 |
| Dual | 0.6317 | 0.4958 | 0.7359 | 0.6119 |

Context selection used the right-arm mapping:

| Context | Context windows | Best validation macro-F1 | Test subset accuracy | Test finger accuracy | Test macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 200 ms | 1 | 0.6414 | 0.4780 | 0.7253 | 0.5366 |
| 500 ms | 4 | 0.6422 | 0.4804 | 0.7438 | 0.5786 |
| 1 s | 9 | 0.6464 | 0.5318 | 0.7585 | 0.5735 |

The selected final policy is right-arm mapping with 9 context windows.

## Five-Seed Direct Training Results

Final direct training used seeds `[0, 1, 2, 3, 4]`, right-arm mapping, 9 context windows, CPU execution, and validation-selected thresholds.

| Metric | Mean | Standard deviation | 95% CI |
| --- | ---: | ---: | ---: |
| Training-selection validation macro-F1 | 0.6480 | 0.0102 | 0.0089 |
| Test subset accuracy | 0.4801 | 0.0080 | 0.0070 |
| Test finger accuracy | 0.7326 | 0.0127 | 0.0111 |
| Test macro-F1 | 0.5822 | 0.0162 | 0.0142 |
| Test macro-AUPRC | 0.6401 | 0.0230 | 0.0202 |
| Test macro-AUROC | 0.7683 | 0.0139 | 0.0122 |

## Final Checkpoint

- Selected seed: 4
- Selection basis: highest training-selection validation macro-F1 among seeds 0-4
- Checkpoint: `cnn_micro_4ch_right_ctx9_seed4.pt`
- Checkpoint SHA-256: `dceff9980141097a2febcf1d9af598eca21f3da6b5b690c1ba02c7a5720e266c`
- Thresholds: `[0.65, 0.60, 0.70, 0.70, 0.60]`

## Distillation Update

Cross-channel distillation has now completed using the full 64-channel teacher representation and the four-channel right-map student representation. The distilled seed-4 checkpoint is the recommended final model because it achieved the highest validation macro-F1 among the final distilled seeds and improved mean test performance over direct training.

| Training mode | Test subset accuracy | Test finger accuracy | Test macro-F1 | Test macro-AUPRC | Test macro-AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct, five seeds | 0.4801 +/- 0.0080 | 0.7326 +/- 0.0127 | 0.5822 +/- 0.0162 | 0.6401 +/- 0.0230 | 0.7683 +/- 0.0139 |
| Distilled, five seeds | 0.5219 +/- 0.0114 | 0.7612 +/- 0.0038 | 0.6095 +/- 0.0058 | 0.6933 +/- 0.0036 | 0.7904 +/- 0.0073 |

Recommended final checkpoint:

- `cnn_micro_4ch_right_ctx9_distill_seed4.pt`
- SHA-256: `3512c065823487dc8b697bb1f0830ef7b0cd61fa5478a7343548a5fe2920d0d8`
