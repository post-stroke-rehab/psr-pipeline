# Mini Outline

- **Abstract:** state the post-stroke finger-intent task, three-model baseline, strongest CNN result, and completed four-channel distillation result.
- **Introduction:** frame the manuscript as the software research layer of a broader neurorehabilitation project and state the study-level contributions.
- **Related Work:** cover post-stroke sEMG rehab, dataset context (Ninapro and PhysioMio), prior stroke decoding studies, deep models, and transfer learning.
- **Method:** formalize notation, preprocessing, shared feature tensor, LSTM/CNN/GNN mathematics, multilabel loss, and the implemented transfer/distillation paths.
- **Experiments:** separate single-run baseline families from five-seed four-channel direct, transfer, and distillation experiments.
- **Results:** present LSTM/CNN/GNN baselines first, then CNN student/teacher improvements, healthy transfer, four-channel distillation, and external positioning.
- **Discussion:** interpret transfer and distillation outcomes, the sensor-density trade-off, and the ONNX/hardware evidence boundary.
- **Conclusion:** identify the distilled four-channel CNN-Micro as the software handoff and Raspberry Pi timing as the next experiment.
