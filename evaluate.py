"""
evaluate.py — Full test-set evaluation for SoilHSI3DCNN.

Runs a held-out dataset through the trained model and reports:
  • Soil health head  : accuracy, macro F1, per-class precision/recall/F1,
                        confusion matrix
  • Contaminant head  : per-contaminant ROC-AUC, F1, precision, recall
                        (at 0.5 threshold), mean AUC
  • Optional plots    : confusion matrix heatmap, ROC curves per contaminant

Usage:
    python evaluate.py --data-dir data/test --labels data/test_labels.csv \
                       --checkpoint runs/best_model.pth

    python evaluate.py --data-dir data/test --labels data/test_labels.csv \
                       --checkpoint runs/best_model.pth --save-plots
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# ── Label definitions (must match train.py) ───────────────────────────────────
HEALTH_LABELS = [
    "Severely Degraded",
    "Degraded",
    "Moderate",
    "Recovering",
    "Healthy / Remediated",
]

CONTAM_NAMES = ["metal", "pfas", "glyphosate", "microplastics"]

CONTAM_THRESHOLD = 0.5   # sigmoid threshold for binary contaminant decision


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_labels_csv(csv_path: str) -> tuple[list[str], list[tuple]]:
    """
    Expects a CSV with columns:
        path, health_class, metal, pfas, glyphosate, microplastics

    Returns:
        paths  : list of .npy file paths
        labels : list of (health_int, [c0, c1, c2, c3]) tuples
    """
    paths, labels = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(row["path"])
            health = int(row["health_class"])
            contam = [float(row[n]) for n in CONTAM_NAMES]
            labels.append((health, contam))
    return paths, labels


# ── Inference loop ────────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Pass the full test set through the model.

    Returns raw arrays:
        health_preds   : (N,)       int   — argmax class
        health_probs   : (N, 5)     float — softmax probabilities
        health_targets : (N,)       int
        contam_probs   : (N, 4)     float — sigmoid probabilities
        contam_targets : (N, 4)     float — ground-truth binary labels
    """
    model.eval()

    all_health_preds   = []
    all_health_probs   = []
    all_health_targets = []
    all_contam_probs   = []
    all_contam_targets = []

    for cube, health, contam in loader:
        cube   = cube.to(device)
        health = health.to(device)
        contam = contam.to(device)

        health_logits, contam_p = model(cube)

        health_p = torch.softmax(health_logits, dim=1)
        preds    = health_p.argmax(dim=1)

        all_health_preds.append(preds.cpu().numpy())
        all_health_probs.append(health_p.cpu().numpy())
        all_health_targets.append(health.cpu().numpy())
        all_contam_probs.append(contam_p.cpu().numpy())
        all_contam_targets.append(contam.cpu().numpy())

    return {
        "health_preds":   np.concatenate(all_health_preds),
        "health_probs":   np.concatenate(all_health_probs),
        "health_targets": np.concatenate(all_health_targets),
        "contam_probs":   np.concatenate(all_contam_probs),
        "contam_targets": np.concatenate(all_contam_targets),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def health_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Accuracy + per-class precision, recall, F1, macro F1."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        confusion_matrix,
    )

    acc     = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    report  = classification_report(
        targets, preds,
        target_names=HEALTH_LABELS,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(targets, preds, labels=list(range(len(HEALTH_LABELS))))

    return {
        "accuracy":  acc,
        "macro_f1":  macro_f1,
        "report":    report,
        "confusion_matrix": cm,
    }


def contam_metrics(probs: np.ndarray, targets: np.ndarray) -> dict:
    """Per-contaminant ROC-AUC, precision, recall, F1 at threshold 0.5."""
    import warnings
    from sklearn.metrics import (
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
        roc_curve,
    )

    preds_bin = (probs >= CONTAM_THRESHOLD).astype(int)
    results   = {}
    auc_list  = []

    for i, name in enumerate(CONTAM_NAMES):
        y_true  = targets[:, i]
        y_score = probs[:, i]
        y_pred  = preds_bin[:, i]

        if y_true.sum() == 0 or (1 - y_true).sum() == 0:
            warnings.warn(
                f"'{name}': only one class present in test set — AUC undefined.",
                RuntimeWarning,
            )
            auc = float("nan")
            fpr = tpr = thresholds = None
        else:
            auc = roc_auc_score(y_true, y_score)
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            auc_list.append(auc)

        results[name] = {
            "auc":       auc,
            "f1":        f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "fpr":       fpr,
            "tpr":       tpr,
        }

    results["mean_auc"] = float(np.nanmean([r["auc"] for r in results.values()
                                             if isinstance(r, dict)]))
    return results


# ── Printing ──────────────────────────────────────────────────────────────────

def print_health_results(m: dict) -> None:
    print("\n" + "═" * 56)
    print("  SOIL HEALTH CLASSIFICATION")
    print("═" * 56)
    print(f"  Accuracy  : {m['accuracy']*100:.2f}%")
    print(f"  Macro F1  : {m['macro_f1']:.4f}")
    print()
    print(f"  {'Class':<26} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
    print("  " + "-" * 50)
    for label in HEALTH_LABELS:
        r = m["report"].get(label, {})
        print(
            f"  {label:<26} "
            f"{r.get('precision', 0):>6.3f} "
            f"{r.get('recall', 0):>6.3f} "
            f"{r.get('f1-score', 0):>6.3f} "
            f"{int(r.get('support', 0)):>5}"
        )

    print("\n  Confusion matrix (rows=true, cols=pred):")
    header = "       " + "".join(f"{i:>5}" for i in range(len(HEALTH_LABELS)))
    print(header)
    for i, row in enumerate(m["confusion_matrix"]):
        print(f"  [{i}]  " + "".join(f"{v:>5}" for v in row))
    print()


def print_contam_results(m: dict) -> None:
    print("═" * 56)
    print("  CONTAMINANT DETECTION  (threshold = 0.50)")
    print("═" * 56)
    print(f"  {'Contaminant':<20} {'AUC':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}")
    print("  " + "-" * 48)
    for name in CONTAM_NAMES:
        r = m[name]
        auc_str = f"{r['auc']:.4f}" if not np.isnan(r["auc"]) else "  NaN "
        print(
            f"  {name:<20} {auc_str:>6} "
            f"{r['f1']:>6.3f} "
            f"{r['precision']:>6.3f} "
            f"{r['recall']:>6.3f}"
        )
    print(f"\n  Mean AUC : {m['mean_auc']:.4f}")
    print("═" * 56 + "\n")


# ── Plots ─────────────────────────────────────────────────────────────────────

def save_confusion_matrix(cm: np.ndarray, output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("⚠️  matplotlib not installed — skipping confusion matrix plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(HEALTH_LABELS)))
    ax.set_yticks(range(len(HEALTH_LABELS)))
    ax.set_xticklabels(HEALTH_LABELS, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(HEALTH_LABELS, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Soil Health — Confusion Matrix")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✅ Confusion matrix saved: {output_path}")
    plt.close()


def save_roc_curves(contam_m: dict, output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not installed — skipping ROC curves.")
        return

    fig, axes = plt.subplots(1, len(CONTAM_NAMES), figsize=(14, 4))

    for ax, name in zip(axes, CONTAM_NAMES):
        r = contam_m[name]
        if r["fpr"] is None:
            ax.text(0.5, 0.5, "AUC undefined\n(single class)",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            ax.plot(r["fpr"], r["tpr"],
                    label=f"AUC = {r['auc']:.3f}", color="#e74c3c", lw=2)
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.legend(loc="lower right", fontsize=9)

        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

    fig.suptitle("Contaminant ROC Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✅ ROC curves saved: {output_path}")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SoilHSI3DCNN on a held-out test set"
    )
    parser.add_argument("--data-dir",   required=True,
                        help="Directory containing test .npy cubes")
    parser.add_argument("--labels",     required=True,
                        help="CSV file with columns: path,health_class,metal,pfas,glyphosate,microplastics")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained .pth checkpoint")
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-plots", action="store_true",
                        help="Save confusion matrix and ROC curve plots")
    parser.add_argument("--output-dir", default="outputs",
                        help="Directory for saved plots (default: outputs/)")
    parser.add_argument("--cpu",        action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() \
             else torch.device("cuda")
    print(f"🖥️  Device: {device}")

    cfg = load_config(args.config)
    num_bands   = cfg["model"]["num_bands"]
    target_size = tuple(cfg["model"]["target_size"])

    # ── Dataset ────────────────────────────────────────────────────────────
    try:
        from dataset import HyperspectralSoilDataset
    except ImportError:
        sys.exit("❌  Cannot import dataset.py. Run from the repo root.")

    paths, labels = load_labels_csv(args.labels)
    test_ds = HyperspectralSoilDataset(
        paths, labels,
        num_bands=num_bands,
        target_size=target_size,
        train=False,           # no augmentation during evaluation
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"✅ Test set: {len(test_ds)} samples")

    # ── Model ──────────────────────────────────────────────────────────────
    try:
        from model import SoilHSI3DCNN
    except ImportError:
        sys.exit("❌  Cannot import model.py. Run from the repo root.")

    model = SoilHSI3DCNN(
        num_bands=num_bands,
        num_classes=len(HEALTH_LABELS),
        num_contaminants=len(CONTAM_NAMES),
    )
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    print(f"✅ Checkpoint loaded: {args.checkpoint}")

    # ── Inference ──────────────────────────────────────────────────────────
    print("Running inference...")
    raw = run_evaluation(model, test_loader, device)

    # ── Metrics ────────────────────────────────────────────────────────────
    h_metrics = health_metrics(raw["health_preds"], raw["health_targets"])
    c_metrics = contam_metrics(raw["contam_probs"], raw["contam_targets"])

    print_health_results(h_metrics)
    print_contam_results(c_metrics)

    # ── Plots ──────────────────────────────────────────────────────────────
    if args.save_plots:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_confusion_matrix(h_metrics["confusion_matrix"],
                               str(out / "confusion_matrix.png"))
        save_roc_curves(c_metrics, str(out / "roc_curves.png"))


if __name__ == "__main__":
    main()
