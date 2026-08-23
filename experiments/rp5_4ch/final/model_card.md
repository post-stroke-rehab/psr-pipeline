# Four-Channel CNN-Micro Model Card

## Model Summary

This checkpoint is a four-active-channel CNN-Micro model for five-finger intent prediction from sEMG. The recommended final model is the distilled seed-4 checkpoint, which improved the five-seed mean test metrics relative to direct four-channel training.

- Architecture: `CNN_Micro`
- Recommended checkpoint: `cnn_micro_4ch_right_ctx9_distill_seed4.pt`
- Checkpoint SHA-256: `3512c065823487dc8b697bb1f0830ef7b0cd61fa5478a7343548a5fe2920d0d8`
- Selection basis: highest training-selection validation macro-F1 across final distilled seeds `[0, 1, 2, 3, 4]`
- Ground/reference channel: excluded from model input
- Completed training modes: direct four-channel training, full64 source training, transfer initialization, and cross-channel distillation
- Transfer learning status: evaluated across five seeds from the compatible full64 `CNNMicroSequence` source checkpoint; it did not outperform direct training or distillation on mean four-channel test metrics

## Intended Use

The model is intended to decode post-stroke finger intent from a reduced set of forearm sEMG channels for a hardware-assisted neurorehabilitation prototype. It provides five independent finger-intent logits that can be consumed by downstream deployment software. Hardware acquisition, ADC firmware, actuator control, safety validation, and real-time integration are outside this model card.

## Input Contract

- Raw active signals before preprocessing: `(samples, 4)`
- Active muscle order: `[ECRB, ECRL, FDS, FDP]`
- Selected channel policy: right-arm map
- Right-arm one-based channels: `[15, 16, 9, 1]`
- Right-arm zero-based channels: `[14, 15, 8, 0]`
- Feature tensor before adaptation: `(batch, 4, windows, 12)`
- CNN input after adaptation: `(batch, windows, 48)`
- Selected context setting: 9 windows
- Output logits: `(batch, 5)`
- Output order: `[thumb, index, middle, ring, little]`
- Validation-selected thresholds: `[0.50, 0.70, 0.60, 0.65, 0.60]`

## Training Data

- Dataset: PhysioMio impaired-arm data
- Sampling rate: 2048 Hz
- Windowing: 200 ms windows with 50% overlap
- Feature count: 12 features per active channel per window
- Patient-level train split: 2656 samples from 34 patients
- Patient-level validation split: 432 samples from 5 patients
- Patient-level test split: 720 samples from 9 patients

## Final Distilled Performance

The final distillation experiment used the selected right-arm channel map and 9-window context across seeds `[0, 1, 2, 3, 4]`. Values below are mean, standard deviation, and 95% confidence interval across those five seeds.

| Metric | Mean | Standard deviation | 95% CI |
| --- | ---: | ---: | ---: |
| Training-selection validation macro-F1 | 0.6824 | 0.0104 | 0.0091 |
| Test subset accuracy | 0.5219 | 0.0114 | 0.0100 |
| Test finger accuracy | 0.7612 | 0.0038 | 0.0034 |
| Test macro-F1 | 0.6095 | 0.0058 | 0.0051 |
| Test macro-AUPRC | 0.6933 | 0.0036 | 0.0031 |
| Test macro-AUROC | 0.7904 | 0.0073 | 0.0064 |

## Comparison Across Training Modes

The selected distilled model outperformed both direct four-channel training and
transfer initialization in the five-seed four-channel comparison. The full64
source model is reported as an upper-reference software baseline because it uses
all 64 PhysioMio HD-sEMG channels rather than the four active hardware channels.

| Training mode | Test subset accuracy | Test finger accuracy | Test macro-F1 | Test macro-AUPRC | Test macro-AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full64 source, seed 42 | 0.5658 | 0.7846 | 0.6810 | 0.7696 | 0.8520 |
| Direct four-channel, five seeds | 0.4801 +/- 0.0080 | 0.7326 +/- 0.0127 | 0.5822 +/- 0.0162 | 0.6401 +/- 0.0230 | 0.7683 +/- 0.0139 |
| Transfer four-channel, five seeds | 0.4618 +/- 0.0089 | 0.7111 +/- 0.0078 | 0.5706 +/- 0.0109 | 0.5985 +/- 0.0164 | 0.7478 +/- 0.0064 |
| Distilled four-channel, five seeds | 0.5219 +/- 0.0114 | 0.7612 +/- 0.0038 | 0.6095 +/- 0.0058 | 0.6933 +/- 0.0036 | 0.7904 +/- 0.0073 |

## Limitations

The model was trained from PhysioMio using anatomical channel maps inferred from planned hardware placement rather than hardware-collected MyoWare recordings. Real deployment performance may change with electrode placement, skin impedance, amplifier characteristics, sampling jitter, and the post-stroke participant population. The checkpoint is ready for software handoff, but hardware-in-the-loop validation remains necessary before rehabilitation use.
