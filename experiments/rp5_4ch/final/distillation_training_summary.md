# Four-Channel CNN-Micro Distillation Summary

## Selected Configuration

- Student model: `CNN_Micro`
- Teacher checkpoint: `results/distill_micro_from_cnn_a0.3_t2.0/checkpoint_best.pt`
- Student active channels: 4
- Student muscle order: `[ECRB, ECRL, FDS, FDP]`
- Student selected channel policy: right-arm mapping
- Right-arm one-based channels: `[15, 16, 9, 1]`
- Right-arm zero-based channels: `[14, 15, 8, 0]`
- Ground/reference channel included: no
- Student CNN input contract: `(batch, windows, 48)`
- Teacher input contract: `(batch, windows, 768)`
- Selected context: 9 windows
- Output order: `[thumb, index, middle, ring, little]`

## Five-Seed Distillation Results

Final distillation used seeds `[0, 1, 2, 3, 4]`, right-arm mapping, 9 context windows, CPU execution, and validation-selected thresholds.

| Metric | Mean | Standard deviation | 95% CI |
| --- | ---: | ---: | ---: |
| Training-selection validation macro-F1 | 0.6824 | 0.0104 | 0.0091 |
| Test subset accuracy | 0.5219 | 0.0114 | 0.0100 |
| Test finger accuracy | 0.7612 | 0.0038 | 0.0034 |
| Test macro-F1 | 0.6095 | 0.0058 | 0.0051 |
| Test macro-AUPRC | 0.6933 | 0.0036 | 0.0031 |
| Test macro-AUROC | 0.7904 | 0.0073 | 0.0064 |

## Direct Versus Distilled

| Training mode | Test subset accuracy | Test finger accuracy | Test macro-F1 | Test macro-AUPRC | Test macro-AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct, five seeds | 0.4801 +/- 0.0080 | 0.7326 +/- 0.0127 | 0.5822 +/- 0.0162 | 0.6401 +/- 0.0230 | 0.7683 +/- 0.0139 |
| Distilled, five seeds | 0.5219 +/- 0.0114 | 0.7612 +/- 0.0038 | 0.6095 +/- 0.0058 | 0.6933 +/- 0.0036 | 0.7904 +/- 0.0073 |

## Final Checkpoint

- Selected seed: 4
- Selection basis: highest training-selection validation macro-F1 among distilled seeds 0-4
- Checkpoint: `cnn_micro_4ch_right_ctx9_distill_seed4.pt`
- Checkpoint SHA-256: `3512c065823487dc8b697bb1f0830ef7b0cd61fa5478a7343548a5fe2920d0d8`
- Thresholds: `[0.50, 0.70, 0.60, 0.65, 0.60]`

## Reload Verification

The final checkpoint was reloaded with `CNNMicroSequence(in_features=48, out_dim=5)` and verified to map a synthetic `(2, 9, 48)` input to `(2, 5)` logits.
