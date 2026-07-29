# Self Review

## 1. Does the paper now tell the requested story?

- **Answer:** Yes. The evaluation begins with three individual model families, LSTM, CNN, and GNN, and only then moves to improvement paths.

## 2. Is the paper written from a research-project perspective rather than as a repository inventory?

- **Answer:** Yes. The framing now centers on a software research framework for post-stroke neurorehabilitation, with the broader hardware goal as context only.

## 3. Is the literature review materially stronger?

- **Answer:** Yes. The manuscript now covers stroke-specific sEMG studies, systematic reviews, Ninapro, PhysioMio, healthy-data deep models, and transfer learning.

## 4. Does the methods section feel technically sound?

- **Answer:** Yes. The revision adds notation, LSTM/CNN/GNN mathematical background, multilabel loss, and the project-specific distillation objective.

## 5. Are the internal comparisons detailed enough?

- **Answer:** Yes. The paper now separates direct baselines from the newer CNN student/teacher branch and includes ResNet teacher results from the merged PR~56 path.

## 6. Is transfer learning handled honestly?

- **Answer:** Yes. The CNN branch is presented as clearly helped by healthy-to-impaired transfer, while the LSTM branch is described as mixed rather than improved.

## 7. Does the paper overclaim distillation or deployment?

- **Answer:** No. Distillation is described as an implemented improvement path, not as a completed benchmark result, and Raspberry Pi deployment remains future work.

## 8. Are external comparisons presented responsibly?

- **Answer:** Yes. The literature-context table explicitly warns that datasets, populations, gestures, and label spaces differ, so it is not presented as a strict leaderboard.

## 9. What is still weakest?

- **Answer:** The project still lacks committed distilled-student metrics, committed device-level latency measurements, and broader robustness evidence across sessions and patients.
