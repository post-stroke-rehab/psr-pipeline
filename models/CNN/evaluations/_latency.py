"""CPU single-sample latency + model size (proxy for Raspberry Pi 5 real-time inference).
Desktop CPU is faster than a Pi 5 A76, but ratios/feasibility carry over."""
import sys, time
from pathlib import Path
import torch
torch.set_num_threads(4)   # Pi 5 has 4 cores
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models" / "CNN"))
from students import CNN_Nano, CNN_Micro, CNN_Base, CNN_Large, CNN_XLarge
from teachers import ResNet50_1D

MODELS = {"nano": CNN_Nano, "micro": CNN_Micro, "base": CNN_Base,
          "large": CNN_Large, "xlarge": CNN_XLarge}

def bench(m, x, iters=100, warmup=20):
    m.eval()
    with torch.no_grad():
        for _ in range(warmup):
            m(x)
        t = []
        for _ in range(iters):
            t0 = time.perf_counter(); m(x); t.append((time.perf_counter()-t0)*1000)
    t.sort()
    return t[len(t)//2], t[int(len(t)*0.95)]

x1 = torch.randn(1, 768, 39)        # single window-segment, real-time scenario
print(f"{'model':<10}{'params':>12}{'fp32_KB':>10}{'lat_ms(b=1)':>13}{'p95_ms':>9}")
for name, cls in MODELS.items():
    m = cls()
    p = sum(t.numel() for t in m.parameters())
    med, p95 = bench(m, x1)
    print(f"{name:<10}{p:>12,}{p*4/1024:>10.0f}{med:>13.3f}{p95:>9.3f}")
m = ResNet50_1D(); p = sum(t.numel() for t in m.parameters())
med, p95 = bench(m, x1, iters=30, warmup=5)
print(f"{'resnet50':<10}{p:>12,}{p*4/1024:>10.0f}{med:>13.3f}{p95:>9.3f}")

# dynamic int8 quantization effect on the Linear layers of nano
m = CNN_Nano().eval()
mq = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
med0, _ = bench(m, x1); medq, _ = bench(mq, x1)
print(f"\nnano fp32 b=1: {med0:.3f} ms   nano dynamic-int8(Linear) b=1: {medq:.3f} ms")
print("note: Conv1d dominates here; dynamic-int8 only quantizes Linear. Static/QAT needed for conv int8.")
