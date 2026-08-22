# Full64 CNN-Micro Source Summary

## Configuration

- Model: `CNN_Micro`
- Run: `source_full64_ctx9_seed42`
- Input contract: `(batch, windows, 768)`
- Feature tensor before adaptation: `(batch, 64, windows, 12)`
- Context: 9 windows
- Output order: `[thumb, index, middle, ring, little]`
- Selection basis: best validation macro-F1 within the source run

## Performance

| Metric | Value |
| --- | ---: |
| Best validation macro-F1 | 0.6976 |
| Test subset accuracy | 0.5658 |
| Test finger accuracy | 0.7846 |
| Test macro-F1 | 0.6810 |
| Test macro-AUPRC | 0.7696 |
| Test macro-AUROC | 0.8520 |

## Role In The Four-Channel Study

This full64 source model is not the hardware deployment target because it depends
on all 64 PhysioMio HD-sEMG channels. It serves two purposes: it provides a
same-split upper-reference software baseline, and it supplies the compatible
`CNNMicroSequence` checkpoint used for first-layer slicing in the four-channel
transfer experiments.
