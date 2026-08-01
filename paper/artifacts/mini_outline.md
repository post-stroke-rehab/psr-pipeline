# Mini Outline

- **Abstract:** state the post-stroke finger-intent task, baseline-three-model story, strongest CNN result, CNN-Micro deployment choice, and transfer asymmetry.
- **Introduction:** frame the manuscript as the software research layer of a broader neurorehabilitation project and state the study-level contributions.
- **Related Work:** cover post-stroke sEMG rehab, dataset context (Ninapro and PhysioMio), prior stroke decoding studies, deep models, and transfer learning.
- **Method:** formalize notation, preprocessing, shared feature tensor, LSTM/CNN/GNN mathematics, multilabel loss, and the implemented transfer/distillation paths.
- **Experiments:** describe patient-level impaired-arm benchmarking, evidence tiers, and why the evaluation is organized as baseline models first and improvement paths second.
- **Results:** present LSTM/CNN/GNN direct baselines first, then CNN student/teacher improvements, then transfer learning and distillation readiness, then finger-wise and literature positioning.
- **Discussion:** explain what the software stack now proves, what transfer learning teaches, why CNN-Micro is the deployment model, and where the work stands relative to prior sEMG research.
- **Conclusion:** restate the software research contribution, identify CNN-Micro as the hardware model, and name distilled-student evaluation plus Raspberry Pi validation as the next evidence milestones.
