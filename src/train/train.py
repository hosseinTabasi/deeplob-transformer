"""Single training entry: YAML config, seed, CPU-friendly tiny defaults."""
from __future__ import annotations
import argparse, csv, random
from pathlib import Path
import numpy as np, torch, yaml
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from data.fi2010 import FI2010Dataset
from models import ControlledTransformer, FusionDeepLOB, LSTMClassifier, MLPClassifier, OFIClassifier

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def build_model(name: str, seq_len: int, n_features: int = 40) -> nn.Module:
    name = name.lower()
    if name == 'mlp': return MLPClassifier(seq_len=seq_len, n_features=n_features)
    if name == 'lstm': return LSTMClassifier(n_features=n_features)
    if name == 'deeplob': return FusionDeepLOB(mode='deeplob')
    if name == 'transformer': return ControlledTransformer(n_features=n_features)
    if name == 'ofi': return OFIClassifier()
    if name == 'both': return FusionDeepLOB(mode='both')
    raise ValueError(f'unknown model {name!r}')

def _macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    total_loss, n = 0.0, 0
    crit = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = crit(logits, yb)
        total_loss += float(loss.item()) * len(yb)
        n += len(yb)
        ys.append(yb.cpu().numpy())
        ps.append(logits.argmax(dim=-1).cpu().numpy())
    y = np.concatenate(ys) if ys else np.array([])
    pred = np.concatenate(ps) if ps else np.array([])
    acc = float((y == pred).mean()) if len(y) else 0.0
    counts = np.bincount(y, minlength=3).tolist() if len(y) else [0, 0, 0]
    return {"loss": total_loss / max(n, 1), "accuracy": acc,
            "macro_f1": _macro_f1(y, pred) if len(y) else 0.0, "n": n,
            "class_counts": counts}

def run_train(cfg, repo_root=None):
    repo_root = Path(repo_root) if repo_root else Path.cwd()
    seed = int(cfg.get("seed", 0)); set_seed(seed)
    seq_len = int(cfg.get("seq_len", 32)); k = int(cfg.get("k", 1))
    protocol = str(cfg.get("split", "paper_setup2"))
    model_name = str(cfg.get("model", "mlp"))
    epochs = int(cfg.get("epochs", 1)); batch_size = int(cfg.get("batch_size", 16))
    lr = float(cfg.get("lr", 1e-3)); device = torch.device("cpu")
    ds = FI2010Dataset.load(root=repo_root, k=k, seq_len=seq_len, seed=seed,
        synthetic_events=int(cfg.get("synthetic_events", 120)),
        synthetic_days=int(cfg.get("synthetic_days", 10)),
        force_synthetic=bool(cfg.get("force_synthetic", False)),
        alpha=float(cfg.get("alpha", 2e-4)))
    masks = ds.split(protocol)
    Xtr, ytr = ds.subset(masks.get("train"))
    va = masks.get("val")
    te = masks.get("test")
    Xte, yte = ds.subset(te)
    split_eval = "test"
    use_va = va is not None and bool(va.any())
    if use_va:
        Xte, yte = ds.subset(va)
        split_eval = "val"
    train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)), batch_size=batch_size, shuffle=False)
    model = build_model(model_name, seq_len=seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    metrics = evaluate(model, eval_loader, device)
    is_toy = ds.source != "fi2010"
    note = "TOY synthetic smoke; not FI-2010; not tradable PnL" if is_toy else "real FI-2010 run"
    row = {
        "label": "TOY" if is_toy else "FI2010",
        "source": ds.source,
        "model": model_name,
        "k": k,
        "split": protocol,
        "eval_split": split_eval,
        "seed": seed,
        "epochs": epochs,
        "n_train": int(len(ytr)),
        "n_eval": int(metrics["n"]),
        "accuracy": round(metrics["accuracy"], 6),
        "macro_f1": round(metrics["macro_f1"], 6),
        "class_counts_eval": metrics["class_counts"],
        "note": note,
    }
    out_dir = repo_root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / str(cfg.get("out_csv", "toy_smoke.csv"))
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    return row

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/toy.yaml")
    args = parser.parse_args()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    root = cfg_path.resolve().parent.parent if cfg_path.exists() else Path.cwd()
    print(run_train(cfg, repo_root=root))

if __name__ == "__main__":
    main()
