"""
AI Accelerator Design - Assignment 3
BCHW  ->  B x (CHW)  ->  BCHW  using explicit index arithmetic (no reshape/view/flatten/ravel).

Address mapping (row-major / C-order, W varies fastest):
    forward : k = c*(H*W) + h*W + w          ,  0 <= k < C*H*W
    reverse : c = k // (H*W)
              h = (k % (H*W)) // W
              w = k % W
The batch index b is unchanged in both directions (it is the row of the 2-D tensor).
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# Core algorithm (from scratch: loops + explicit indexing only)
# ---------------------------------------------------------------------------
def flatten_bchw(I, B, C, H, W):
    """B x C x H x W  ->  B x (C*H*W)"""
    CHW, HW = C * H * W, H * W
    F = np.empty((B, CHW), dtype=I.dtype)          # allocate only
    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    k = c * HW + h * W + w         # linear address inside sample b
                    F[b, k] = I[b, c, h, w]
    return F


def reconstruct_bchw(F, B, C, H, W):
    """B x (C*H*W)  ->  B x C x H x W  (index generated from k)"""
    CHW, HW = C * H * W, H * W
    R = np.empty((B, C, H, W), dtype=F.dtype)      # allocate only
    for b in range(B):
        for k in range(CHW):
            c = k // HW
            rem = k % HW
            h = rem // W
            w = rem % W
            R[b, c, h, w] = F[b, k]
    return R


def errors(I, R):
    E = np.abs(I.astype(np.float64) - R.astype(np.float64))
    return float(E.max()), float(E.mean())


# ---------------------------------------------------------------------------
# Manual indexing demo for the small case B=1, C=2, H=2, W=3
# ---------------------------------------------------------------------------
def manual_demo():
    B, C, H, W = 1, 2, 2, 3
    I = np.empty((B, C, H, W), dtype=np.int64)
    for c in range(C):
        for h in range(H):
            for w in range(W):
                I[0, c, h, w] = 10 * (c + 1) + 3 * h + w   # readable distinct values
    F = flatten_bchw(I, B, C, H, W)
    print("Manual example  B=1, C=2, H=2, W=3  (k = c*HW + h*W + w, HW=6, W=3)")
    print(f"{'(b,c,h,w)':>12} {'k':>4}  value")
    for c in range(C):
        for h in range(H):
            for w in range(W):
                k = c * H * W + h * W + w
                print(f"{str((0, c, h, w)):>12} {k:>4}  {I[0, c, h, w]}")
    print("Flattened row 0:", F[0].tolist())
    R = reconstruct_bchw(F, B, C, H, W)
    print("Reconstruction exact:", errors(I, R) == (0.0, 0.0))
    print()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(name, B):
    """Return a B x C x H x W float32 array. Uses torchvision if available,
    otherwise falls back to random data of the same shape and range."""
    try:
        import torchvision
        import torchvision.transforms as T
        cls = {"MNIST": torchvision.datasets.MNIST,
               "CIFAR-10": torchvision.datasets.CIFAR10}[name]
        ds = cls(root="./data", train=False, download=True, transform=T.ToTensor())
        imgs = [ds[i][0].numpy() for i in range(B)]       # each is C x H x W
        out = np.empty((B,) + imgs[0].shape, dtype=np.float32)
        for b in range(B):
            out[b] = imgs[b]                               # stack by assignment
        return out, name
    except Exception:
        shape = {"MNIST": (B, 1, 28, 28), "CIFAR-10": (B, 3, 32, 32)}[name]
        rng = np.random.default_rng(0)
        return rng.random(shape, dtype=np.float32), name + " (synthetic stand-in)"


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def run():
    manual_demo()
    rng = np.random.default_rng(42)
    cases = []
    B = 4
    for name in ("MNIST", "CIFAR-10"):
        I, label = load_dataset(name, B)
        cases.append((label, I))
    for i, C in enumerate([8, 16, 32, 64, 128, 256, 500], start=1):
        H = W = 16
        I = rng.standard_normal((B, C, H, W)).astype(np.float32)   # synthetic feature map
        cases.append((f"Synthetic fmap{i}", I))

    print(f"{'Dataset/Input':<30}{'B':>3}{'C':>5}{'H':>4}{'W':>4}"
          f"{'CHW':>9}{'Emax':>8}{'MAE':>8}{'time(s)':>9}")
    rows = []
    for label, I in cases:
        B_, C, H, W = I.shape
        t0 = time.time()
        F = flatten_bchw(I, B_, C, H, W)
        R = reconstruct_bchw(F, B_, C, H, W)
        dt = time.time() - t0
        emax, mae = errors(I, R)
        rows.append((label, B_, C, H, W, emax, mae))
        print(f"{label:<30}{B_:>3}{C:>5}{H:>4}{W:>4}{C*H*W:>9}{emax:>8.1f}{mae:>8.1f}{dt:>9.2f}")

    # -----------------------------------------------------------------------
    # OPTIONAL REFERENCE CHECK (NOT part of the algorithm): compare against
    # NumPy's reshape purely as an independent verification.
    # -----------------------------------------------------------------------
    ok = all(np.array_equal(flatten_bchw(I, *I.shape), I.reshape(I.shape[0], -1))
             for _, I in cases[:3])
    print("\n[reference check, numpy reshape] flatten matches:", ok)
    return rows


if __name__ == "__main__":
    run()
