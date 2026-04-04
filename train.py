import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import warnings

from model import SoilHSI3DCNN
from dataset import HyperspectralSoilDataset

from model import SpectralCNN 


# ── Config ────────────────────────────────────────────────────────────────────

CONTAMINANT_NAMES = ["metal", "pfas", "glyphosate", "microplastics"]

CFG = dict(
    num_bands        = 200,
    num_classes      = 5,
    num_contaminants = 4,
    batch_size       = 16,
    epochs           = 100,
    lr               = 3e-4,
    weight_decay     = 0.05,
    # FIX 1 — explicit loss weight so CE and BCE scales are independently tunable.
    # BCE loss is typically smaller in magnitude than CE; start at 0.5 and tune
    # on a held-out set if one task dominates the gradient.
    contam_loss_weight = 0.5,
    label_smoothing  = 0.1,
    # CosineAnnealingWarmRestarts: first restart after T_0 epochs, then T_0*T_mult, …
    cosine_T0        = 10,
    cosine_T_mult    = 2,
    grad_clip        = 1.0,
    num_workers      = 8,
    save_path        = "soil_3dcnn_enotrium.pth",
)


# ── Loss helpers ──────────────────────────────────────────────────────────────

def compute_loss(
    pred_health: torch.Tensor,
    pred_contam: torch.Tensor,
    health: torch.Tensor,
    contam: torch.Tensor,
    ce_criterion: nn.CrossEntropyLoss,
    contam_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (total_loss, ce_loss, bce_loss).

    FIX 1 — losses are kept separate so per-task magnitudes can be logged
    and the weighting hyperparameter `contam_weight` is explicit.
    """
    ce_loss  = ce_criterion(pred_health, health)
    bce_loss = F.binary_cross_entropy(pred_contam, contam)
    total    = ce_loss + contam_weight * bce_loss
    return total, ce_loss, bce_loss


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    ce_criterion: nn.CrossEntropyLoss,
    contam_weight: float,
    contam_names: list[str],
    device: torch.device,
) -> dict:
    """
    FIX 4 — full validation pass returning epoch-mean losses and per-class
    ROC-AUC scores for the contaminant head.

    Returns a dict with keys:
        val_loss, val_ce_loss, val_bce_loss,
        val_acc,
        auc_<contaminant_name> for each contaminant,
        val_auc_mean
    """
    model.eval()

    total_loss = total_ce = total_bce = 0.0
    n_batches = 0
    correct = 0
    total_samples = 0

    all_contam_probs  = []   # (N, num_contaminants)
    all_contam_labels = []   # (N, num_contaminants)

    for cube, health, contam in loader:
        cube, health, contam = cube.to(device), health.to(device), contam.to(device)

        pred_health, pred_contam = model(cube)

        loss, ce_loss, bce_loss = compute_loss(
            pred_health, pred_contam, health, contam, ce_criterion, contam_weight
        )

        total_loss += loss.item()
        total_ce   += ce_loss.item()
        total_bce  += bce_loss.item()
        n_batches  += 1

        preds = pred_health.argmax(dim=1)
        correct       += (preds == health).sum().item()
        total_samples += health.size(0)

        all_contam_probs.append(pred_contam.cpu().numpy())
        all_contam_labels.append(contam.cpu().numpy())

    # Epoch-mean losses
    metrics = {
        "val_loss"    : total_loss / n_batches,
        "val_ce_loss" : total_ce   / n_batches,
        "val_bce_loss": total_bce  / n_batches,
        "val_acc"     : correct / total_samples,
    }

    # FIX 4 — per-class ROC-AUC for the contaminant head.
    # roc_auc_score requires at least one positive sample per class; fall back
    # gracefully with a warning when a class is absent in the validation split.
    probs  = np.concatenate(all_contam_probs,  axis=0)   # (N, C)
    labels = np.concatenate(all_contam_labels, axis=0)   # (N, C)

    auc_scores = []
    for c, name in enumerate(contam_names):
        y_true = labels[:, c]
        y_score = probs[:, c]
        if y_true.sum() == 0 or (1 - y_true).sum() == 0:
            warnings.warn(
                f"Contaminant '{name}' has only one class in the validation "
                f"split — AUC is undefined; skipping.",
                RuntimeWarning,
            )
            metrics[f"auc_{name}"] = float("nan")
        else:
            auc = roc_auc_score(y_true, y_score)
            metrics[f"auc_{name}"] = auc
            auc_scores.append(auc)

    metrics["val_auc_mean"] = float(np.mean(auc_scores)) if auc_scores else float("nan")
    return metrics


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Replace with real paths / labels
    data_paths: list = [...]
    labels:     list = [...]

    train_paths, val_paths, train_lbl, val_lbl = train_test_split(
        data_paths, labels, test_size=0.2, random_state=42
    )

    train_ds = HyperspectralSoilDataset(train_paths, train_lbl, train=True)
    val_ds   = HyperspectralSoilDataset(val_paths,   val_lbl,   train=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True
    )

    model = SoilHSI3DCNN(
        num_bands=cfg["num_bands"],
        num_classes=cfg["num_classes"],
        num_contaminants=cfg["num_contaminants"],
    ).to(device)

    optimizer  = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # FIX 2 — scheduler is stepped per-batch with a fractional epoch value so
    # warm restarts fire at the correct iteration count, not just once per epoch.
    scheduler  = CosineAnnealingWarmRestarts(optimizer, T_0=cfg["cosine_T0"], T_mult=cfg["cosine_T_mult"])
    scaler     = GradScaler()
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    best_auc = -1.0

    for epoch in range(cfg["epochs"]):
        model.train()

        # FIX 3 — running accumulators for epoch-mean loss tracking
        running_loss = running_ce = running_bce = 0.0
        n_batches = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch:03d} [train]", leave=False)

        for i, (cube, health, contam) in pbar:
            cube    = cube.to(device, non_blocking=True)
            health  = health.to(device, non_blocking=True)
            contam  = contam.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                pred_health, pred_contam = model(cube)
                # FIX 1 — weighted loss; contam_loss_weight is a tunable scalar
                loss, ce_loss, bce_loss = compute_loss(
                    pred_health, pred_contam, health, contam,
                    ce_criterion, cfg["contam_loss_weight"]
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            # FIX 2 — step scheduler with fractional epoch so CosineAnnealingWarmRestarts
            # interpolates correctly within the epoch, not just at epoch boundaries.
            scheduler.step(epoch + i / len(train_loader))

            # FIX 3 — accumulate for running mean, not last-batch snapshot
            running_loss += loss.item()
            running_ce   += ce_loss.item()
            running_bce  += bce_loss.item()
            n_batches    += 1

            pbar.set_postfix(
                loss=f"{running_loss / n_batches:.4f}",
                ce=f"{running_ce   / n_batches:.4f}",
                bce=f"{running_bce  / n_batches:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        # Epoch-mean train losses
        mean_loss = running_loss / n_batches
        mean_ce   = running_ce   / n_batches
        mean_bce  = running_bce  / n_batches

        # FIX 4 — full validation with per-class ROC-AUC
        val_metrics = validate(
            model, val_loader, ce_criterion,
            cfg["contam_loss_weight"], CONTAMINANT_NAMES, device
        )

        # ── logging ───────────────────────────────────────────────────────────
        auc_str = "  ".join(
            f"{name}={val_metrics[f'auc_{name}']:.3f}"
            for name in CONTAMINANT_NAMES
        )
        print(
            f"Epoch {epoch:03d} | "
            f"train loss {mean_loss:.4f} (ce {mean_ce:.4f} bce {mean_bce:.4f}) | "
            f"val loss {val_metrics['val_loss']:.4f}  acc {val_metrics['val_acc']:.3f} | "
            f"AUC mean {val_metrics['val_auc_mean']:.3f}  [{auc_str}]"
        )

        # ── checkpoint on best mean AUC ───────────────────────────────────────
        if val_metrics["val_auc_mean"] > best_auc:
            best_auc = val_metrics["val_auc_mean"]
            torch.save(model.state_dict(), cfg["save_path"])
            print(f"  ↳ saved best model  (mean AUC {best_auc:.4f})")

    print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")
    print(f"Model saved to: {cfg['save_path']}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(CFG)
