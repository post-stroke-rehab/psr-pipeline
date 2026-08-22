# Four-Channel CNN-Micro Transfer Summary

## Source Checkpoint

- Source model: full64 `CNN_Micro`
- Source run: `source_full64_ctx9_seed42`
- Source input contract: `(batch, windows, 768)`
- Source active channels: 64 PhysioMio HD-sEMG channels
- Source best validation macro-F1: `0.6976`
- Source test macro-F1: `0.6810`
- Source checkpoint SHA-256: `6edd2202c590fddb8e230a76e1a38f9510a6f20701739808ce888a61b0325c73`

The source checkpoint is treated as a full-channel software reference and as the
initialization point for four-channel transfer. Its checkpoint remains an
intermediate run artifact, while the final committed deployment checkpoint remains
the distilled four-channel model.

## Transfer Method

The transfer runs initialize the four-channel `CNN_Micro` from the full64 source
checkpoint by slicing the first projection layer to the selected right-arm feature
indices. Later compatible layers are loaded directly. The student input contract is
`(batch, windows, 48)` and the output remains five logits ordered
`[thumb, index, middle, ring, little]`.

## Five-Seed Transfer Results

Final transfer used seeds `[0, 1, 2, 3, 4]`, right-arm mapping, 9 context windows,
CPU execution, and validation-selected thresholds.

| Metric | Mean | Standard deviation | 95% CI |
| --- | ---: | ---: | ---: |
| Training-selection validation macro-F1 | 0.6487 | 0.0136 | 0.0119 |
| Test subset accuracy | 0.4618 | 0.0089 | 0.0078 |
| Test finger accuracy | 0.7111 | 0.0078 | 0.0068 |
| Test macro-F1 | 0.5706 | 0.0109 | 0.0095 |
| Test macro-AUPRC | 0.5985 | 0.0164 | 0.0144 |
| Test macro-AUROC | 0.7478 | 0.0064 | 0.0056 |

## Interpretation

Transfer learning completed successfully but did not improve the reduced-channel
CNN-Micro in this experiment. Its mean test macro-F1 was below both direct
four-channel training (`0.5822 +/- 0.0162`) and cross-channel distillation
(`0.6095 +/- 0.0058`). The final deployment handoff therefore remains the distilled
four-channel checkpoint.
