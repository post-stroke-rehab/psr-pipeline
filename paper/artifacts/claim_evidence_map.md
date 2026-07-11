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

Claim: `This manuscript is a systems benchmark paper, not a state-of-the-art algorithm paper.`  
Evidence: benchmark coverage is limited to current repo artifacts; no stored ablation package or cross-dataset evaluation is present.  
Status: supported
