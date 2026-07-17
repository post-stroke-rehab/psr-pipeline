import torch, numpy as np
from collections import Counter

P = "datasets/processed"
out = {}
for split in ["train", "val", "test"]:
    d = torch.load(f"{P}/{split}.pt", map_location="cpu", weights_only=True)
    X, y, meta = d["X"].float(), d["y"].float(), d.get("meta", {})
    N, C, W, F = X.shape
    # padding: a window is "padded" if it is exactly all-zero across all channels & features
    win_allzero = (X.abs().sum(dim=(1, 3)) == 0)            # (N, W) True where window is all zeros
    frac_pad = win_allzero.float().mean().item()
    real_w = (~win_allzero).sum(dim=1)                       # real (non-pad) windows per sample
    # label vectors
    labs = [tuple(int(v) for v in row) for row in (y > 0.5).int().tolist()]
    cnt = Counter(labs)
    finger_pos = (y > 0.5).float().mean(0).tolist()
    print(f"\n==== {split}  N={N} C={C} W={W} F={F} ====")
    print(f"  meta.patients (n={len(meta.get('patients', []))}): {meta.get('patients')}")
    print(f"  frac of windows that are all-zero padding: {frac_pad:.3f}")
    print(f"  real windows/sample: min={real_w.min().item()} med={int(real_w.median())} max={real_w.max().item()} mean={real_w.float().mean():.1f}")
    print(f"  per-finger positive rate [thumb,index,mid,ring,little]: {[round(v,3) for v in finger_pos]}")
    print(f"  unique label vectors ({len(cnt)}):")
    for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
        print(f"     {k}: {v}  ({100*v/N:.1f}%)")
    out[split] = (N, C, W, F, frac_pad, cnt)

# feature scale check on train: stats per feature index across all (real) windows
d = torch.load(f"{P}/train.pt", map_location="cpu", weights_only=True)
X = d["X"].float()
N, C, W, F = X.shape
fnames = ["RMS","MAV","IEMG","WL","VAR","ZC","SSC","WAMP","MNF","MDF","SEN","TP"]
# mask out padded windows
win_real = (X.abs().sum(dim=(1,3)) != 0)   # (N,W)
Xr = X.permute(0,2,1,3).reshape(N*W, C, F)[win_real.reshape(-1)]   # (n_real, C, F)
print("\n==== TRAIN feature scale (over real windows & channels) ====")
print(f"{'feat':<6}{'mean':>12}{'std':>12}{'min':>12}{'max':>12}")
for i, fn in enumerate(fnames):
    col = Xr[:, :, i].reshape(-1)
    print(f"{fn:<6}{col.mean().item():>12.4g}{col.std().item():>12.4g}{col.min().item():>12.4g}{col.max().item():>12.4g}")

# how separable is rest vs active from a simple energy feature? (sanity)
y = d["y"].float()
rest_mask = (y > 0.5).sum(1) == 0
# mean RMS per sample over real windows+channels
rms_idx = 0
samp_rms = []
for n in range(N):
    wr = win_real[n]
    samp_rms.append(X[n][:, wr, rms_idx].mean().item() if wr.any() else 0.0)
samp_rms = torch.tensor(samp_rms)
print(f"\nmean per-sample RMS: rest={samp_rms[rest_mask].mean():.4g} (n={rest_mask.sum()})  active={samp_rms[~rest_mask].mean():.4g} (n={(~rest_mask).sum()})")
print(f"  rest RMS quartiles: {np.percentile(samp_rms[rest_mask].numpy(),[25,50,75])}")
print(f"  active RMS quartiles: {np.percentile(samp_rms[~rest_mask].numpy(),[25,50,75])}")
