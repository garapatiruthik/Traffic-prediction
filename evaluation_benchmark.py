"""
evaluation_benchmark.py — Step 3: Inference & 3-model benchmark
Models: Naive-Repeat | MambaProxyFFN-A (no weather) | MambaProxyFFN-B (weather)
"""
import os, sys, json, time as _time
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import random as _rn
import numpy as np
np.random.seed(42)
_rn.seed(42)

import torch
torch.manual_seed(42)
torch.set_num_threads(8)

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

import seaborn as sns

PROCESSED_DIR = Path("./data/processed")
RESULTS_DIR   = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

LOOKBACK  = 24
HORIZON   = 12
NUM_BINS  = 256


# ── baseline ─────────────────────────────────────────────────────────────────

class NaiveRepeat(torch.nn.Module):
    """Repeats last observed speed over the horizon."""
    def __init__(self, horizon=HORIZON): super().__init__(); self.h=horizon
    def forward(self, x):
        return x[:, -1, 0:1].repeat(1, self.h)


# ── MambaProxyFFN (mirrors proxy_ablation_training.py) ───────────────────────

class MambaProxyFFN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, horizon=HORIZON, dropout=0.20):
        super().__init__()
        self.flat_dim = input_dim * LOOKBACK
        self.l1 = torch.nn.Sequential(torch.nn.Linear(self.flat_dim, hidden_dim),
                       torch.nn.LayerNorm(hidden_dim), torch.nn.GELU(), torch.nn.Dropout(dropout))
        self.l2 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim),
                       torch.nn.LayerNorm(hidden_dim), torch.nn.GELU(), torch.nn.Dropout(dropout))
        self.head = torch.nn.Linear(hidden_dim, horizon)
    def forward(self, x):
        B = x.shape[0]
        return self.head(self.l2(self.l1(x.reshape(B, -1))))


# ── metric helpers ────────────────────────────────────────────────────────────

def mae(pred, true): return np.mean(np.abs(pred - true))

def rmse(pred, true): return np.sqrt(np.mean((pred - true) ** 2))

def kl_div(pred, true, k=NUM_BINS, eps=1e-9):
    """Histogram-based symmetric KL divergence for 1D arrays (evaluation axis)."""
    lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
    edges = np.linspace(lo, hi, k + 1)
    ph, _  = np.histogram(pred,  edges, density=True)
    gt, _  = np.histogram(true,  edges, density=True)
    ph = np.clip(ph, eps, 1.0);  gt = np.clip(gt, eps, 1.0)
    ph /= ph.sum();               gt /= gt.sum()
    return 0.5 * (np.sum(gt * np.log(gt / ph)) + np.sum(ph * np.log(ph / gt)))


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, X_tensor):
    device = torch.device("cpu")
    model.eval().to(device)
    dl = DataLoader(TensorDataset(X_tensor), batch_size=4096, shuffle=False, num_workers=0)
    out = []
    for (xb,) in dl:
        out.append(model(xb.float()).cpu().numpy())
    return np.concatenate(out, axis=0)


def evaluate(name, y_pred_scaled, y_true_scaled, mean, std, scale_name):
    inv_pred = y_pred_scaled * std + mean
    inv_true = y_true_scaled * std + mean
    e_mae = mae(inv_pred, inv_true)
    e_rmse = rmse(inv_pred, inv_true)
    e_kl  = kl_div(inv_pred.flatten(), inv_true.flatten())
    return {f"{scale_name}_MAE":  round(e_mae, 4),
            f"{scale_name}_RMSE": round(e_rmse, 4),
            f"{scale_name}_KL":   round(e_kl,  6)}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("="*60 + "\nEVALUATION BENCHMARK\n" + "="*60)

    # ── load data ──
    data = torch.load(PROCESSED_DIR / "processed_data.pt", map_location="cpu", weights_only=False)
    sp   = np.load(PROCESSED_DIR / "scaler_params.npz", allow_pickle=True)
    t_mean, t_std = float(sp["traffic_mean"]), float(sp["traffic_std"])

    X_A, y_A = data["X_test_A"], data["y_test_A"]
    X_B, y_B = data["X_test_B"], data["y_test_B"]

    N, L, DA = X_A.shape
    _, _, DB = X_B.shape
    print(f"  Test windows: {N}  |  Dims: A={DA} B={DB}  |  Horizon={HORIZON}")

    model_names = ["Naive", "Model_A", "Model_B"]
    results     = {}

    # ═══════════════════════════════════════════
    # 1. Naive-Repeat
    # ═══════════════════════════════════════════
    print("\n── Inference [Naive-Repeat] ──")
    t0 = _time.monotonic()
    naive = NaiveRepeat()
    p_naive = naive(torch.tensor(X_A))
    t_lat = (_time.monotonic() - t0) * 1000 / N * 1000   # ms per sample
    metrics_n = evaluate("Naive", p_naive.numpy(), y_A, t_mean, t_std, "scaled")
    metrics_n["latency_ms"] = round(t_lat, 3)
    results["Naive"] = metrics_n
    print(f"  MAE={metrics_n['scaled_MAE']:.4f}  RMSE={metrics_n['scaled_RMSE']:.4f}  "
          f"KL={metrics_n['scaled_KL']:.6f}  latency={t_lat:.2f}ms")

    # ═══════════════════════════════════════════
    # 2. MambaProxyFFN-A  (no weather)
    # ═══════════════════════════════════════════
    print("\n── Inference [Model_A] ──")
    ckpt_A = Path("./checkpoints/model_A_best.pt")
    if not ckpt_A.exists():
        print("  [WARN] checkpoint not found — skipping Model_A")
    else:
        mA = MambaProxyFFN(input_dim=DA)
        mA.load_state_dict(torch.load(ckpt_A, weights_only=True))
        t0 = _time.monotonic()
        p_A = run_inference(mA, torch.tensor(X_A))
        t_lat = (_time.monotonic() - t0) * 1000 / N * 1000
        metrics_A = evaluate("Model_A", p_A, y_A, t_mean, t_std, "scaled")
        metrics_A["latency_ms"] = round(t_lat, 3)
        results["Model_A"] = metrics_A
        print(f"  MAE={metrics_A['scaled_MAE']:.4f}  RMSE={metrics_A['scaled_RMSE']:.4f}  "
              f"KL={metrics_A['scaled_KL']:.6f}  latency={t_lat:.2f}ms")

    # ═══════════════════════════════════════════
    # 3. MambaProxyFFN-B  (weather)
    # ═══════════════════════════════════════════
    print("\n── Inference [Model_B] ──")
    ckpt_B = Path("./checkpoints/model_B_best.pt")
    if not ckpt_B.exists():
        print("  [WARN] checkpoint not found — skipping Model_B")
    else:
        mB = MambaProxyFFN(input_dim=DB)
        mB.load_state_dict(torch.load(ckpt_B, weights_only=True))
        t0 = _time.monotonic()
        p_B = run_inference(mB, torch.tensor(X_B))
        t_lat = (_time.monotonic() - t0) * 1000 / N * 1000
        metrics_B = evaluate("Model_B", p_B, y_B, t_mean, t_std, "scaled")
        metrics_B["latency_ms"] = round(t_lat, 3)
        results["Model_B"] = metrics_B
        print(f"  MAE={metrics_B['scaled_MAE']:.4f}  RMSE={metrics_B['scaled_RMSE']:.4f}  "
              f"KL={metrics_B['scaled_KL']:.6f}  latency={t_lat:.2f}ms")

    # ═══════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════
    print("\n" + "="*70)
    hdr = f"{'Model':<12} {'MAE':>8} {'RMSE':>8} {'KL_256':>9} {'Lat_ms':>8}"
    print(hdr); print("-"*70)
    for nm in model_names:
        if nm in results:
            r = results[nm]
            print(f"{nm:<12} {r['scaled_MAE']:>8.4f} {r['scaled_RMSE']:>8.4f} "
                  f"{r['scaled_KL']:>9.6f} {r.get('latency_ms','N/A'):>8.2f}")
    print("="*70)

    summary = {
        "setup": {"N_windows": int(N), "Lookback": LOOKBACK, "Horizon": HORIZON,
                  "Num_Bins_KL": NUM_BINS},
        "models": results
    }
    out_path = RESULTS_DIR / "evaluation_results.json"
    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"\n[OK] Results saved → {out_path}")


if __name__ == "__main__":
    main()
