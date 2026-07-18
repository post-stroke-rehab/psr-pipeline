import torch, numpy as np
from sklearn.metrics import f1_score

yte = torch.load("datasets/processed/test.pt", map_location="cpu", weights_only=True)["y"]
yte = (yte > 0.5).float().numpy()
ytr = torch.load("datasets/processed/train.pt", map_location="cpu", weights_only=True)["y"]
ytr = (ytr > 0.5).float().numpy()
N, C = yte.shape

def report(name, pred):
    exact = (pred == yte).all(1).mean()
    fingacc = (pred == yte).mean()
    f1 = f1_score(yte, pred, average="macro", zero_division=0)
    print(f"{name:<28} exact={exact*100:5.2f}  fingacc={fingacc*100:5.2f}  f1_macro={f1*100:5.2f}")

print(f"test N={N}")
report("all-zeros (rest)", np.zeros_like(yte))
report("all-ones", np.ones_like(yte))
# per-finger majority from train
maj = (ytr.mean(0) > 0.5).astype(float)
report(f"per-finger majority {maj.tolist()}", np.tile(maj, (N, 1)))
# most-frequent label vector from train
from collections import Counter
cnt = Counter(tuple(r) for r in ytr.astype(int).tolist())
top = np.array(max(cnt, key=cnt.get), float)
report(f"most-freq vector {top.tolist()}", np.tile(top, (N, 1)))
# random per training marginals
rng = np.random.default_rng(0)
p = ytr.mean(0)
report("sample~train marginals", (rng.random((N, C)) < p).astype(float))
