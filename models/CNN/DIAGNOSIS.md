# CNN Pipeline Diagnosis & Experiments

_EMG → 5-finger flexion, PhysioMio impaired-arm, cross-subject. Prepared for the Sunday meeting._

All numbers below are measured on the **test set** in a clean, controlled harness
(`models/CNN/evaluations/_harness.py` + `_experiments.py`), 60 epochs, AdamW+cosine,
best-val-F1 checkpoint. Where you see `±`, it is **mean ± std over 5 random seeds**.

---

## TL;DR (read this first)

1. **The ceiling is the DATA, not the models.** ~44% of all samples are *active muscle
   contractions mislabeled as "rest."* A dumb "always predict rest" baseline already
   gets **66% finger-accuracy / 50% exact-match**. Our models (~80% / ~59%) are only
   ~14 / ~9 points above that floor. No architecture can fix a labeling problem.

2. **The smallest model (nano, 54K params) is the best and most stable.** No larger
   model beats it beyond noise. The **ResNet teachers (16M params) are *worse* than nano** —
   they overfit the ~34 training patients and don't transfer.

3. **Knowledge distillation cannot help here.** You can't distill an advantage the teacher
   doesn't have. Tested both framings × 5 seeds: **no gain** (f1 within noise, AUPRC slightly worse).

4. **Run-to-run training noise (±1–3.5 f1) is larger than the gap between model sizes.**
   The "base 79% vs nano 78%" ranking we've been chasing is **noise**, not signal.

5. **Real-time on a Pi 5 is NOT the binding constraint.** *Every* model — even the 16M
   ResNet — fits the 100 ms inference budget. nano has ~40× headroom. So the small model
   isn't a compromise we're forced into; it's the genuinely better choice. Quantization
   isn't needed for speed.

6. **The one real win I found: ensemble 5 nano models** → best result in the study
   (f1 **74.9**, AUPRC **84.0**, finger-acc **82.1**), still tiny (272K params, real-time).

---

## PART A — Issues the TEAM owns (not in `models/CNN/`, needs discussion)

### A1. ⚠️ Label mapping collapses active gestures into "rest" — **the biggest problem**

`data_processing/mapping.py` maps **8 distinct movements to `[0,0,0,0,0]`**:
`Rest, PinchGrasp, MassExtension, MassAdduction, WristVolarFlexion, WristDorsiFlexion,
ForearmPronation, ForearmSupination`. Seven of those are **active contractions.**

**Proof it's systematic, not a rounding artifact** — the label counts in the processed
data are *exact* multiples of 231 (one gesture's sample count):

| label vector | train count | = | gestures feeding it |
|---|---|---|---|
| `[0,0,0,0,0]` | 1848 | 8 × 231 | rest + 7 active movements |
| `[1,1,1,1,1]` | 693 | 3 × 231 | MassFlexion, DiameterGrasp, SphereGrasp |
| `[0,1,1,1,1]` | 231 | 1 × 231 | HookGrasp |
| `[1,1,0,0,0]` | 231 | 1 × 231 | ThumbAdduction |
| `[1,0,1,0,0]` | 231 | 1 × 231 | PinchGraspMiddle |
| `[1,0,0,1,0]` | 231 | 1 × 231 | PinchGraspRing |
| `[1,0,0,0,1]` | 231 | 1 × 231 | PinchGraspPinkie |

So **only 1/16 of "rest" is true rest; ~44% of the entire dataset is active movement
labeled as nothing.** We are literally training the model to call strong extension/wrist/
forearm EMG "no fingers moving." This contaminates the dominant class and caps everyone.

**Root cause:** the 5-bit finger-*flexion* target physically *cannot* represent extension,
wrist deviation, or forearm rotation — they have no valid code, so `gesture_to_5bit`
defaults them to `[0,0,0,0,0]`.

**Likely bug:** `PinchGrasp → DB2 id 15 = "wrist radial deviation" = [0,0,0,0,0]`. A pinch
should be at least thumb+index. (PinchGraspMiddle/Ring/Pinkie *are* mapped correctly; the
plain PinchGrasp is not.)

**Why we can't fix it in `models/CNN/`:** the dataloader saves **only the 5-bit label** and
throws away the gesture name. Downstream, a mislabeled-rest sample is *indistinguishable*
from a true-rest sample. The fix must happen upstream.

**Options to decide as a team:**
- (a) **Drop the un-representable gestures** before training so "rest" means rest. Cleanest;
  costs data but removes the contradiction.
- (b) **Change the target to gesture classification** (predict 1 of ~16 gestures), and derive
  fingers downstream. Uses all data honestly. Needs gesture id saved.
- (c) **Expand the label space** (add extension / wrist DOFs) — bigger product change.
- (d) Minimum viable: **fix the PinchGrasp mapping** and **save `gesture_id` + `patient_id`**
  in the processed tensors so we can even measure this.

### A2. Cross-subject split + tiny validation set make every comparison noisy
- 48 patients → 70/10/20 → ~34 train / **~5 val** / ~10 test patients; 3696 train samples.
- A 5-patient val set drives model selection and HPO — that's the main source of the
  ±1–3.5 f1 noise. EMG amplitude varies wildly across subjects; transfer is the hard part.
- **Asks:** save `patient_id`; use **k-fold cross-subject CV**; **report mean±std over
  seeds/folds**, never a single run.

### A3. Input is hand-engineered features, not raw signal (the "input dimensionality" question)
- The CNN never sees raw EMG. `preprocess_emg` produces **(64 ch × 39 windows × 12 features)**
  — RMS/MAV/IEMG/WL/VAR/ZC/SSC/WAMP/MNF/MDF/SEN/TP per 200 ms window.
- These 12 aggregates are a strong, lossy bottleneck (waveform/phase is gone). This is *also*
  why depth/width don't help: the signal is already summarized, so there's little extra
  structure for capacity to exploit.
- **Discuss:** is the feature pipeline the right representation, or should we try a model on
  **raw windows** (`PreprocessConfig.return_windows=True` already exists)? This is the
  dimensionality decision you flagged — it's a team call because it changes the shared contract.

---

## PART B — What I tried INSIDE `models/CNN/` and the outcomes

| Approach | params | f1@0.5 | AUPRC | exact-acc | verdict |
|---|---|---|---|---|---|
| trivial "always rest" | 0 | 0.0 | — | 50.0 | floor (finger-acc 66.3) |
| **nano** (multilabel) | 54K | **73.0 ± 1.2** | **83.2 ± 0.8** | 61.3 ± 1.4 | **best single, tightest** |
| nano (multiclass, 7-way) | 54K | 71.8 ± 0.7 | 81.0 ± 1.0 | 61.5 ± 1.7 | most stable; good for KD |
| nano + feature standardization | 54K | 70.8 ± 2.8 | 78.1 ± 3.4 | 58.2 ± 2.8 | **hurts** |
| nano + amplitude augmentation | 54K | 70.4 ± 1.1 | 79.3 ± 1.4 | 58.1 ± 1.7 | **hurts** |
| micro | 158K | 71.4 ± 2.6 | 81.1 ± 4.3 | 59.3 ± 3.6 | ≈ nano, noisier |
| base | 390K | 72.0 ± 2.2 | 82.3 ± 2.4 | 61.6 ± 2.6 | ≈ nano, noisier |
| large / xlarge | 0.8–1.6M | ~71–73 | ~78–81 | — | within noise, not better |
| Teacher ResNet50 (multilabel) | 16.3M | 65–73 (high var) | 71–80 | — | **worse than nano, overfits** |
| Teacher ResNet50 (multiclass) | 16.3M | 68–70 | 76 | — | **worse than nano** |
| KD: nano ← teacher (multilabel) | 54K | 73.5 ± 1.1 | 80.8 ± 1.7 | — | **no gain** (≈ plain nano) |
| KD: nano ← teacher (multiclass) | 54K | 71.5 ± 1.5 | 79.6 ± 0.8 | — | **no gain** |
| **ENSEMBLE: 5 × nano** | 272K | **74.9** | **84.0** | **63.3** | **best overall (finger-acc 82.1, AUROC 90.3)** |

**Reading the table:**
- **Normalization / standardization hurts** (both per-feature z-score and InstanceNorm). The
  *absolute amplitude* of the features is the rest-vs-active cue; normalizing erases it. → keep `in_norm=False`.
- **Amplitude augmentation hurts** for the same reason (scale-jitter perturbs the cue). Don't use it.
- **Threshold tuning** (per-finger, tuned on val) adds only ~1 f1 for students (their precision/recall
  are already balanced ~67/78). It helps the *teacher's* f1 more (its recall is low) but never lifts it
  above nano — so the teacher's weakness is **real, not just a 0.5-threshold artifact.**
- **Bigger is not better.** Every larger student sits inside nano's noise band, with *more* variance.
- **The teacher genuinely underperforms the student even on threshold-free AUPRC/AUROC** — classic
  cross-subject overfitting (it can hit ~98% train but doesn't transfer). Teachers are also very
  high-variance run to run.
- **KD has no headroom** because the teacher ≤ student. Both framings (multilabel & multiclass),
  5 seeds each at alpha 0.5: no gain. Structurally guaranteed — distillation can't add signal the
  teacher doesn't have.
- **Ensembling 5 nano seeds is the one real, deployable win** — it directly cancels the per-seed noise
  that's the dominant source of instability, and it's still tiny and real-time.

---

## PART C — A measurement bug in my own HPO that muddied the size comparison

`models/CNN/optimization.py` scores each trial as **`f1 × speed_penalty(latency_ratio)`**.
The penalty drops the score hard for any config slower than that size's baseline, so within
each size class HPO is pushed toward *faster (smaller)* configs. Combined with single-run
noise, the `optuna_*` numbers in `evaluations/summary.json` are **not** a clean "does capacity
help" comparison — they conflate accuracy with speed and sit inside the noise band anyway.

**Fix:** since real-time isn't binding (Part D), **drop the speed penalty** from the accuracy
search (or replace it with a hard latency *cap*), select on **mean val-F1 over seeds**, and
report mean±std.

---

## PART D — Real-time on Raspberry Pi 5 (and why KD/quantization aren't needed for speed)

Single-sample (batch=1) CPU latency, 4 threads (a Pi 5 A76 is ~4–6× slower than this desktop):

| model | params | size (fp32) | latency b=1 | Pi-5 estimate | vs 100 ms budget |
|---|---|---|---|---|---|
| nano | 54K | 212 KB | 0.49 ms | ~2–3 ms | **~40× real-time** |
| base | 390K | 1.5 MB | 1.24 ms | ~6–8 ms | ~13× |
| xlarge | 1.6M | 6.3 MB | 2.29 ms | ~10–14 ms | ~8× |
| ResNet50 teacher | 16.3M | 64 MB | 12.1 ms | ~50–70 ms | still < 100 ms |
| **ensemble 5× nano** | 272K | 1.0 MB | ~2.5 ms | ~12–15 ms | ~7× |

Inference cadence is 10 Hz (100 ms window step). **Everything fits**, including the ResNet.
So:
- The small model is **not** a latency compromise — it's chosen because it **generalizes
  better** (less overfitting). That's a stronger story for the paper than "we had to shrink it."
- **Dynamic int8 quantization made nano *slower*** (the conv layers dominate, and dynamic
  quant only touches `Linear`). Quantization would only matter for power/thermal, not speed,
  and would need static/QAT on the conv layers.
- **KD's premise doesn't hold here.** KD pays off when you're forced onto a weak small model
  and want to recover a strong teacher's accuracy. Here the small model is already real-time
  *and* better, and the teacher is worse. The KD code is correct and ready — **shelve it until
  the data is fixed and a genuinely stronger teacher exists.**

---

## PART E — Recommendations & priorities

**For the team / lead (highest leverage, upstream of `models/CNN/`):**
1. **Fix the labels.** Stop collapsing active gestures into rest. Pick a target representation
   (drop un-representable gestures, or switch to gesture-classification). Fix the PinchGrasp
   mapping. **This is the only thing that raises the ceiling.**
2. **Save `patient_id` and `gesture_id`** in the processed tensors so we can do per-subject
   normalization and even *measure* the contamination.
3. **Adopt k-fold cross-subject evaluation** and report **mean ± std**, not single runs. A
   ±2 f1 "improvement" on one seed is noise.
4. **Decide the input representation** (engineered features vs raw windows) — the dimensionality
   question. Worth a small bake-off once labels are clean.

**For me / the CNN (inside `models/CNN/`, ready now):**
5. **Ship nano** as the deployment model — best, most stable, ~40× real-time. Offer the
   **5× nano ensemble** as the higher-accuracy option (still real-time).
6. **Drop the HPO speed penalty**; select on mean val-F1 over seeds.
7. **Keep `in_norm=False`; no amplitude augmentation.** Both hurt because amplitude is signal.
8. **Shelve KD** (no headroom on current data); keep the code for after the labels are fixed.

**Honest framing for the meeting:** we built solid teacher/student/KD/HPO machinery, and the
experiments are conclusive — but they show the bottleneck is **data quality (labels + cross-
subject + lossy features)**, not model capacity. The most "noble" engineering (a big teacher,
KD) can't beat a 54K-param CNN here, because there's no extra signal to transfer. Fix the
labels and these same tools become worth re-running.

---

_Reproduce: `python models/CNN/evaluations/_experiments.py {variance|capacity|teacher|kd2|extra}`;
data facts: `_diag_data.py`; baselines: `_baselines.py`; latency: `_latency.py`._


# Convo

  The single most important finding

  ~44% of all your training data is active muscle movement mislabeled as "rest." data_processing/mapping.py collapses 8 distinct gestures into
  [0,0,0,0,0] — and 7 of them are active (PinchGrasp, MassExtension, MassAdduction, both wrist flexions, both forearm rotations). I proved it's
  systematic: the label counts are exact multiples of 231 (one gesture's sample count) — rest = 8×231, all-ones = 3×231, the other five = 1×231 each.
  A dumb "always predict rest" classifier already scores 66% finger-accuracy / 50% exact-match; your models at ~80%/59% are only ~14/~9 points above
  that floor. No architecture can out-train a label that says strong extension EMG is "nothing."

  The root cause is conceptual: a 5-bit finger-flexion target literally cannot represent extension, wrist, or forearm gestures, so they silently
  default to all-zeros. And critically — the dataloader saves only the 5-bit label and discards the gesture name, so this contamination is invisible
  and unfixable inside models/CNN/. It has to be fixed upstream. That's your #1 talking point.

  Answering your three goals directly, with evidence

  - "I need the teacher to be incredible." It can't be, on this data. The 16M-param ResNet50 is worse than the 54K nano even on threshold-free
  AUPRC/AUROC (71–80 vs 83). With only ~34 training patients it overfits subjects and fails to transfer. It's not a tuning miss — more capacity has
  nothing left to learn because the features are pre-aggregated and the labels are contaminated.
  - "I need KD with real improvement." Tested both framings (multilabel + 7-way multiclass), 5 seeds each: no gain — within noise on f1, slightly
  worse on AUPRC. This is structural: you can't distill an edge the teacher doesn't have. KD is the right tool for a different situation (a strong
  teacher + a forced-weak student); your code is correct, so shelve it until the labels are fixed and a genuinely better teacher exists.
  - "Real-time on a Pi 5." Not a constraint at all. nano runs at ~0.5 ms (batch=1) — ~40× the 10 Hz budget; even the ResNet fits under 100 ms. Dynamic
  int8 actually made nano slower (conv dominates). So the small model isn't a compromise — it's the better choice because it generalizes more. That's
  a stronger paper story than "we had to shrink it."

  What I tried in your folder, and what actually worked

  The one real win: an ensemble of 5 nano models → best result in the whole study (f1 74.9, AUPRC 84.0, finger-acc 82.1), still 272K params and
  real-time. It works by canceling per-seed noise. Everything else either matched nano within noise (micro/base/large/xlarge) or hurt (feature
  standardization and amplitude augmentation both erase the absolute-amplitude cue, which is exactly the rest-vs-active signal).

  The meta-finding that changes how the team should work

  Run-to-run training noise is ±1–3.5 f1 — larger than the gap between any of your model sizes. The "base 79% vs nano 78%" ranking you've been chasing
  is noise, not signal (I reproduced the identical config giving 67%, 72%, 67% across runs). Two consequences: report mean±std over seeds, never
  single runs; and your HPO objective in optimization.py multiplies f1 by a speed penalty, which quietly biases the size comparison — drop it, since
  latency isn't binding.

  For the Sunday meeting (team-owned, upstream of your code)

  1. Fix the labels — stop collapsing active gestures into rest; pick a target (drop the un-representable gestures, or switch to
  gesture-classification). Fix the PinchGrasp → [0,0,0,0,0] mapping bug. This is the only thing that raises the ceiling.
  2. Save patient_id + gesture_id in the processed tensors (enables per-subject normalization and even measuring the contamination).
  3. k-fold cross-subject CV + mean±std reporting (your ~5-patient val set is why selection is noisy).
  4. The input-dimensionality question you raised: you're feeding 12 hand-engineered features, not raw signal — worth a bake-off (raw windows vs
  features) after labels are clean.

  I saved these conclusions to memory and left the experiment scripts in models/CNN/evaluations/ so you (or I) can re-run any of it with python
  models/CNN/evaluations/_experiments.py {variance|capacity|teacher|kd2|extra}.

✻ Crunched for 3m 21s

✻ Claude resuming /loop wakeup (Jun 11 2:21am)

● This is the stale fallback timer I set earlier — but the work it points to is already complete: I read the extra results (ensemble f1 74.9 / AUPRC
  84.0; augmentation hurts), wrote models/CNN/DIAGNOSIS.md, updated memory, and delivered the full summary. Nothing to redo.

  Let me just confirm no stray GPU processes are lingering from my background runs, for cleanliness.

● Bash(tasklist 2>/dev/null | grep -ic python; echo "report exists:"; ls -la models/CNN/DIAGNOSIS.md 2>/dev/null | awk '{print $5, $9}')
  ⎿  0
     report exists:
     12577 models/CNN/DIAGNOSIS.md
  ⎿  Allowed by auto mode classifier

● Everything's clean: 0 stray Python processes (the hung KD process was killed; the extra run exited cleanly via the hard-exit safeguard), and the
  report is in place at models/CNN/DIAGNOSIS.md (12.5 KB).

  This was a stale fallback wakeup — the diagnosis is already finished and delivered. To recap what's done, nothing pending:

  - Report written → models/CNN/DIAGNOSIS.md (structured for your Sunday meeting: data issues the team owns, my in-folder experiments with outcomes,
  the HPO confound, real-time/Pi-5 analysis, and prioritized recommendations).
  - Reproducible harness → models/CNN/evaluations/ (_experiments.py, _harness.py, _diag_data.py, _baselines.py, _latency.py).
  - Memory updated → new cnn-experiment-conclusions entry plus sharpened diagnosis notes.

  The bottom line for your meeting: the pipeline is well-built, but the experiments conclusively show the ceiling is data quality (44% mislabeled
  "rest," cross-subject split, lossy pre-aggregated features) — not model size. That's why bigger models, the teacher, and KD all fail to beat a
  54K-param nano. The two actionable wins are the 5×nano ensemble (best result, still real-time) on your side, and fixing the label mapping + saving
  patient/gesture ids on the team's side, which is the only thing that raises the ceiling and makes KD worth revisiting.

✻ Brewed for 25s

※ recap: Goal: make the CNN finger-classifier (and KD) actually work. The diagnosis is complete and the report is written to models/CNN/DIAGNOSIS.md.
  Next: review that report with your team, prioritizing the label-mapping fix.
