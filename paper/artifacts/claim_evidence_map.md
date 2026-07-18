# Claim-Evidence Map

Claim: `PSR-Pipeline packages data harmonization, preprocessing, feature extraction, tensor adaptation, model comparison, transfer-learning support, and metric reporting in one repository.`  
Evidence: `README.md`, `datasets/`, `data_processing/`, `adapters/`, `training/`, `models/`, `evaluation/`, committed metric files.  
Status: supported

Claim: `The default preprocessing pipeline applies a 20--450 Hz zero-phase Butterworth filter, wavelet denoising with sym4/4 levels/soft universal thresholding, and 200 ms windows with 50% overlap.`  
Evidence: `data_processing/preprocess_config.py`, `README.md`.  
Status: supported

Claim: `Each channel-window pair is summarized by 12 handcrafted features spanning time and frequency domains.`  
Evidence: `README.md`, `data_processing/feature_extraction.py`.  
Status: supported

Claim: `The shared adapter converts standardized feature tensors into (N, W, C×F) sequences used across LSTM, CNN, and GNN branches.`  
Evidence: `adapters/feature_to_sequence.py`, `training/train.py`.  
Status: supported

Claim: `The paper's updated system figure surfaces data harmonization steps such as label alignment, patient-level splitting, tensor caching, threshold tuning, and latency benchmarking that are implemented across the repository workflow.`  
Evidence: `datasets/loaders.py`, `data_processing/`, `training/train_distill.py`, `scripts/eval_thresholds.py`, `scripts/benchmark_student_latency.py`, committed figure asset `paper/figures/pipeline_overview.png`.  
Status: supported

Claim: `The data loader uses patient-level 70/10/20 splitting to reduce train-test leakage.`  
Evidence: `datasets/loaders.py`.  
Status: supported

Claim: `The default PhysioMio loading path targets impaired-arm recordings unless reconfigured.`  
Evidence: `datasets/loaders.py` (`arm_split="impaired"`, `impaired_only=True`).  
Status: supported

Claim: `The merged Optuna CNN-Large artifact is the strongest committed benchmark result in the current repository.`  
Evidence: `models/CNN/evaluations/summary.json`, `models/CNN/evaluations/optuna_large/metrics.json`, compared against `metrics/lstm/metrics.json`, `metrics/gnn/test/metrics.json`, and `metrics/cnn/metrics.json`.  
Status: supported

Claim: `Compact CNN students remain competitive while staying within small INT8 footprints intended for Raspberry Pi 5-class deployment.`  
Evidence: `models/CNN/README.md`, `models/CNN/evaluations/summary.json`, per-model metrics under `models/CNN/evaluations/*/metrics.json`.  
Status: supported

Claim: `Teacher ResNet evaluations provide a stronger CNN reference line than the older direct baseline.`  
Evidence: `models/CNN/evaluations/teacher_resnet50/metrics.json`, `teacher_resnet101/metrics.json`, `teacher_resnet152/metrics.json`.  
Status: supported

Claim: `Healthy-to-impaired transfer learning helps the CNN branch more clearly than the LSTM branch in the committed summaries.`  
Evidence: `training/tuning/cnn/both_stages_summary.json`, `training/tuning/lstm/both_stages_summary.json`.  
Status: supported

Claim: `The repository now includes teacher-ensemble distillation, per-finger threshold tuning, and CPU latency benchmarking utilities.`  
Evidence: `training/train_distill.py`, `training/teacher_ensemble.py`, `scripts/eval_thresholds.py`, `scripts/benchmark_student_latency.py`.  
Status: supported

Claim: `The repository does not yet commit a complete distilled-student evaluation bundle or Raspberry Pi 5 on-device timing artifact.`  
Evidence: search across `models/CNN/evaluations/`, `results/`, `scripts/`, and repository root shows code for these utilities but no matching committed result pack or Raspberry Pi benchmark log.  
Status: supported

Claim: `Hardware integration and real-time intent output are project goals, but this manuscript is limited to the software stack.`  
Evidence: manuscript framing in `sections/introduction.tex`, `sections/method.tex`, `sections/discussion.tex`; project context from user and poster is used only as motivation, not as quantitative evidence.  
Status: supported

Claim: `This manuscript is a software systems paper, not a claim of clinical readiness or definitive state-of-the-art superiority.`  
Evidence: benchmark coverage is limited to current repo artifacts; no stored ablation package, cross-dataset evaluation, or clinical validation is present.  
Status: supported
