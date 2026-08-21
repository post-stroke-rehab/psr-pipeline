# Four-Channel CNN-Micro Model Card

## Intended Use

This model predicts five finger-intent logits from four active forearm sEMG channels ordered as `[ECRB, ECRL, FDS, FDP]`. It is intended as the software model for a Raspberry Pi 5 post-stroke neurorehabilitation prototype. Hardware acquisition, ADC firmware, actuator control, and closed-loop safety validation are handled outside this repository.

## Inputs and Outputs

- Raw active input before preprocessing: `(samples, 4)`
- Feature tensor: `(batch, 4, windows, 12)`
- CNN input: `(batch, windows, 48)`
- Output: five logits ordered `[thumb, index, middle, ring, little]`
- Ground/reference: not a model input

## Training Data

- Dataset: PhysioMio
- Sampling rate: `2048 Hz`
- Windowing: `200 ms`, `50%` overlap
- Channel-map policy: left-map, right-map, or dual-map, as recorded in the run manifest
- Split policy: patient-level train/validation/test split, recorded in split manifests

## Model Selection

Select the final checkpoint by mean validation macro-F1 across seeds. Test metrics should be reported only after selecting the mapping/context/policy from validation results.

## Reported Metrics

The final report should include subset accuracy, finger accuracy, macro precision, macro recall, macro-F1, AUPRC, AUROC, per-finger metrics, mean, standard deviation, and 95% confidence intervals across five seeds.

## Known Limitations

The four-channel hardware placement is an intentionally reduced input setting compared with the 64-channel PhysioMio representation. PhysioMio laterality metadata does not directly identify which anatomical map applies to each patient, so left-map and right-map views are tracked explicitly.
