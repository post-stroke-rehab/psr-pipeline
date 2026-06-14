import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import (load_data, get, train, train_kd, full_eval, pr, set_seed, discover_classes,
                      STUDENTS, TEACHERS, n_params, DEV)
import torch

stage = sys.argv[1] if len(sys.argv) > 1 else "core"
t0 = time.time()
load_data()
print(f"[data loaded in {time.time()-t0:.1f}s on {DEV}]")


def run_student(size, standardize, task="multilabel", epochs=60, lr=1e-3, wd=1e-4, bs=32, tag=None, seed=42, quiet=False, aug=None):
    set_seed(seed)
    Xtr, ytr = get("train", standardize); Xva, yva = get("val", standardize)
    classes = discover_classes(ytr) if task == "multiclass" else None
    nc = classes.size(0) if classes is not None else 5
    model = STUDENTS[size](num_classes=nc)
    t = time.time()
    model, vf1 = train(model, Xtr, ytr, Xva, yva, epochs=epochs, lr=lr, wd=wd, bs=bs,
                       task=task, classes=classes, aug=aug)
    dt = time.time() - t
    tag = tag or f"{size} std={int(standardize)} task={task}"
    res = full_eval(model, standardize=standardize, task=task, classes=classes, tag=tag)
    res["sec"] = dt; res["params"] = n_params(model)
    if not quiet:
        pr(res)
        print(f"     (val_f1={vf1*100:.2f}  {dt:.0f}s  {res['params']:,} params)")
    return res, model


def multi(size, standardize, task="multilabel", seeds=range(5), **kw):
    import numpy as np
    rs = [run_student(size, standardize, task=task, seed=s, quiet=True, **kw)[0] for s in seeds]
    def stat(k):
        v = np.array([r[k] for r in rs]) * 100
        return v.mean(), v.std()
    f1m, f1s = stat("f1@0.5"); am, as_ = stat("auprc"); accm, accs = stat("acc@0.5")
    tag = f"{size} std={int(standardize)} task={task}"
    print(f"{tag:<30} f1@0.5={f1m:5.2f}+-{f1s:4.2f}  auprc={am:5.2f}+-{as_:4.2f}  "
          f"acc={accm:5.2f}+-{accs:4.2f}  (n={len(rs)}, raw f1={[round(r['f1@0.5']*100,1) for r in rs]})")
    return rs


def run_teacher(tname, standardize=False, task="multilabel", epochs=40, lr=1e-3, wd=1e-4, bs=32, tag=None):
    set_seed(42)
    Xtr, ytr = get("train", standardize); Xva, yva = get("val", standardize)
    classes = discover_classes(ytr) if task == "multiclass" else None
    nc = classes.size(0) if classes is not None else 5
    model = TEACHERS[tname](num_classes=nc)
    t = time.time()
    model, vf1 = train(model, Xtr, ytr, Xva, yva, epochs=epochs, lr=lr, wd=wd, bs=bs,
                       task=task, classes=classes)
    dt = time.time() - t
    tag = tag or f"TEACHER {tname} std={int(standardize)} task={task}"
    res = full_eval(model, standardize=standardize, task=task, classes=classes, tag=tag)
    res["sec"] = dt; res["params"] = n_params(model)
    pr(res)
    print(f"     (val_f1={vf1*100:.2f}  {dt:.0f}s  {res['params']:,} params)")
    return res, model


def run_kd(size, teacher_model, standardize, task="multilabel", epochs=60, T=4.0, alpha=0.5, tag=None):
    set_seed(42)
    Xtr, ytr = get("train", standardize); Xva, yva = get("val", standardize)
    classes = discover_classes(ytr) if task == "multiclass" else None
    nc = classes.size(0) if classes is not None else 5
    student = STUDENTS[size](num_classes=nc)
    t = time.time()
    student, vf1 = train_kd(student, teacher_model, Xtr, ytr, Xva, yva, epochs=epochs,
                            task=task, classes=classes, T=T, alpha=alpha)
    dt = time.time() - t
    tag = tag or f"KD {size}<-teacher std={int(standardize)} task={task} T={T} a={alpha}"
    res = full_eval(student, standardize=standardize, task=task, classes=classes, tag=tag)
    res["sec"] = dt; res["params"] = n_params(student)
    pr(res)
    print(f"     (val_f1={vf1*100:.2f}  {dt:.0f}s)")
    return res, student


if stage == "extra":
    import numpy as np
    from _harness import get as _get, ensemble_probs, full_eval, predict_probs
    from evaluation.metrics import compute_multilabel_metrics
    import torch
    print("\n=== EXTRA: augmentation + ensembling (attack cross-subject variance) ===")
    print("ref: plain nano ML f1=73.0+-1.2 auprc=83.2+-0.8")
    # amplitude/channel augmentation, 5 seeds
    multi("nano", standardize=False, aug={"scale_jitter": 0.3, "channel_dropout": 0.1})
    # ensemble of 5 plain nano seeds: train, average test probs
    models = []
    for s in range(5):
        _, m = run_student("nano", standardize=False, seed=s, quiet=True)
        models.append(m)
    Xte, yte = _get("test", False)
    yte_np = (yte > 0.5).float().cpu().numpy()
    tp = ensemble_probs(models, Xte)
    m05 = compute_multilabel_metrics(torch.tensor(tp), torch.tensor(yte_np), threshold=0.5)
    print(f"ENSEMBLE(5x nano) std=0 ml      f1@0.5={m05['f1_macro']*100:5.2f}  acc={m05['accuracy']*100:5.2f}  "
          f"fingacc={m05['finger_accuracy']*100:5.2f}  auprc={m05['auprc_macro']*100:5.2f}  auroc={m05['auroc_macro']*100:5.2f}")

if stage == "variance":
    print("\n=== VARIANCE / does anything beat nano beyond noise? (5 seeds each) ===")
    multi("nano",  standardize=False)
    multi("nano",  standardize=True)
    multi("micro", standardize=False)
    multi("base",  standardize=False)
    multi("nano",  standardize=False, task="multiclass")

if stage == "core":
    print("\n=== CORE: nano, standardization x threshold (multilabel) ===")
    run_student("nano", standardize=False)
    run_student("nano", standardize=True)

if stage == "teacher":
    print("\n=== TEACHER quality + threshold tuning (the 'teacher is bad' claim) ===")
    run_teacher("resnet50", standardize=False, task="multilabel")
    run_teacher("resnet50", standardize=False, task="multiclass")

if stage == "kd2":
    import numpy as np
    def multi_kd(teacher_model, task, seeds=range(5), T=4.0, alpha=0.5):
        Xtr, ytr = get("train", False); Xva, yva = get("val", False)
        classes = discover_classes(ytr) if task == "multiclass" else None
        rs = []
        for s in seeds:
            set_seed(s)
            from _harness import train_kd, full_eval as fe
            nc = classes.size(0) if classes is not None else 5
            st = STUDENTS["nano"](num_classes=nc)
            st, _ = train_kd(st, teacher_model, Xtr, ytr, Xva, yva, epochs=60,
                             task=task, classes=classes, T=T, alpha=alpha)
            rs.append(fe(st, standardize=False, task=task, classes=classes, tag=""))
        f1 = np.array([r["f1@0.5"] for r in rs]) * 100
        ap = np.array([r["auprc"] for r in rs]) * 100
        print(f"KD nano<-teacher task={task} T={T} a={alpha}:  "
              f"f1@0.5={f1.mean():5.2f}+-{f1.std():4.2f}  auprc={ap.mean():5.2f}+-{ap.std():4.2f}  "
              f"(raw f1={[round(x,1) for x in f1]})")
    print("\n=== KD (definitive, teacher trained once, 5 student seeds) ===")
    print("ref: plain nano ML f1=73.0+-1.2 auprc=83.2+-0.8 ; plain nano MC f1=71.8+-0.7 auprc=81.0+-1.0")
    print("[training multilabel teacher]")
    _, t_ml = run_teacher("resnet50", standardize=False, task="multilabel", tag="  teacher(ml)")
    multi_kd(t_ml, "multilabel", alpha=0.5)
    print("[training multiclass teacher]")
    _, t_mc = run_teacher("resnet50", standardize=False, task="multiclass", tag="  teacher(mc)")
    multi_kd(t_mc, "multiclass", alpha=0.5)
    multi_kd(t_mc, "multiclass", alpha=0.3)   # lean harder on soft targets (KD's best shot)

if stage == "kd":
    print("\n=== KD: does a good teacher help the nano student? ===")
    print("[multilabel] train teacher then distill ...")
    _, t_ml = run_teacher("resnet50", standardize=False, task="multilabel", tag="  teacher(ml) ref")
    run_student("nano", standardize=False, task="multilabel", tag="  nano plain (ml) ref")
    run_kd("nano", t_ml, standardize=False, task="multilabel")
    print("[multiclass] train teacher then distill ...")
    _, t_mc = run_teacher("resnet50", standardize=False, task="multiclass", tag="  teacher(mc) ref")
    run_student("nano", standardize=False, task="multiclass", tag="  nano plain (mc) ref")
    run_kd("nano", t_mc, standardize=False, task="multiclass")

if stage == "task":
    print("\n=== TASK FRAMING: multilabel vs multiclass (nano, standardized) ===")
    run_student("nano", standardize=True, task="multilabel")
    run_student("nano", standardize=True, task="multiclass")

if stage == "capacity":
    print("\n=== CAPACITY (standardized, multilabel, raw f1, no speed penalty) ===")
    for size in ["nano", "micro", "base", "large", "xlarge"]:
        run_student(size, standardize=True, task="multilabel")

print(f"\n[stage '{stage}' done in {time.time()-t0:.0f}s]")
import os; os._exit(0)   # avoid CUDA-on-Windows exit-cleanup deadlock (all output already flushed via -u)
