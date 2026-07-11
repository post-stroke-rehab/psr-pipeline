# Claim-Evidence Map

Claim: `PSR-Pipeline packages preprocessing, feature extraction, tensor adaptation, model comparison, and metric reporting in one repository.`  
Evidence: `README.md`, `data_processing/`, `adapters/`, `training/train.py`, `evaluation/`, `metrics/`.  
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

Claim: `The paper's updated system figure surfaces data harmonization steps such as label alignment, patient-level splitting, and tensor caching that are implemented across the repository workflow.`  
Evidence: `datasets/loaders.py`, `data_processing/`, `README.md`, committed figure asset `paper/figures/pipeline_overview.png`.  
Status: supported

Claim: `The data loader uses patient-level 70/10/20 splitting to reduce train-test leakage.`  
Evidence: `datasets/loaders.py`.  
Status: supported

Claim: `The default PhysioMio loading path targets impaired-arm recordings unless reconfigured.`  
Evidence: `datasets/loaders.py` (`arm_split="impaired"`, `impaired_only=True`).  
Status: supported

Claim: `Among the saved benchmark artifacts, the LSTM achieves the best exact-match accuracy, finger accuracy, and macro AUROC.`  
Evidence: `metrics/lstm/metrics.json`, `metrics/cnn/metrics.json`, `metrics/gnn/test/metrics.json`.  
Status: supported

Claim: `Among the saved benchmark artifacts, the GNN achieves the best macro F1 and macro AUPRC.`  
Evidence: `metrics/gnn/test/metrics.json`, `metrics/lstm/metrics.json`, `metrics/cnn/metrics.json`.  
Status: supported

Claim: `The GNN is strongest on the thumb label, while the LSTM is strongest on the remaining four finger labels by F1.`  
Evidence: per-finger entries in `metrics/gnn/test/metrics.json`, `metrics/lstm/metrics.json`, `metrics/cnn/metrics.json`.  
Status: supported

Claim: `The repository contains two-stage healthy-to-impaired training artifacts, but they are summarized separately from the main benchmark tables.`  
Evidence: `scripts/run_two_stage.py`, `results/README.md`.  
Status: supported

Claim: `Open PR #56 reorganizes the CNN branch into modular student, teacher, training, distillation, and optimization stages around explicit 1D input reshaping.`  
Evidence: PR `#56` description and changed-file patches (`models/CNN/README.md`, `models/CNN/optimization.py`).  
Status: supported as open-PR evidence

Claim: `Open PR #59 adds ensemble distillation, per-finger threshold tuning, and local CPU latency benchmarking, and reports a provisional CNN-Micro distilled checkpoint that improves on the stated CNN baseline.`  
Evidence: PR `#59` description and changed-file patches (`training/train_distill.py`, `training/teacher_ensemble.py`, `scripts/eval_thresholds.py`, `scripts/benchmark_student_latency.py`).  
Status: supported as open-PR evidence

Claim: `Open PR details are treated in the paper as pending extensions rather than merged benchmark evidence.`  
Evidence: manuscript framing in `sections/abstract.tex`, `sections/experiments.tex`, `sections/results.tex`, `sections/discussion.tex`.  
Status: supported

Claim: `This manuscript is a systems benchmark paper, not a state-of-the-art algorithm paper.`  
Evidence: benchmark coverage is limited to current repo artifacts; no stored ablation package or cross-dataset evaluation is present.  
Status: supported
