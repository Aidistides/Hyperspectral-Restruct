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
import argparse
import json
import os

from model import SoilHSI3DCNN
from dataset import HyperspectralSoilDataset 


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

def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: torch.optim.Optimizer, 
                   scheduler: torch.optim.lr_scheduler._LRScheduler, device: torch.device):
    """Load training checkpoint and return starting epoch and best_auc"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_auc = checkpoint['best_auc']
        print(f"Resumed training from epoch {start_epoch}, best AUC: {best_auc:.4f}")
        return start_epoch, best_auc
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return 0, -1.0


def train(cfg: dict, resume_path: str = None):
    """Main training function with optional resume capability"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {device}")
        
        # Validate configuration
        required_keys = ["num_bands", "num_classes", "num_contaminants", "batch_size", 
                         "epochs", "lr", "weight_decay", "contam_loss_weight", 
                         "label_smoothing", "cosine_T0", "cosine_T_mult", "grad_clip", 
                         "num_workers", "save_path"]
        
        for key in required_keys:
            if key not in cfg:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # For now, create dummy data - replace with actual data loading
        print("Warning: Using dummy data - replace with actual data loading")
        num_samples = 100
        
        # Create dummy hyperspectral data (batch_size, bands, height, width)
        dummy_data = torch.randn(num_samples, cfg["num_bands"], 32, 32)
        labels = torch.randint(0, cfg["num_classes"], (num_samples,))
        contaminant_labels = torch.randint(0, 2, (num_samples, cfg["num_contaminants"])).float()
        
        # Split data
        train_indices, val_indices = train_test_split(
            range(num_samples), test_size=0.2, random_state=42
        )
        
        train_data = dummy_data[train_indices]
        val_data = dummy_data[val_indices]
        train_health = labels[train_indices]
        val_health = labels[val_indices]
        train_contam = contaminant_labels[train_indices]
        val_contam = contaminant_labels[val_indices]

        # Create simple dataset class for dummy data
        class DummyDataset:
            def __init__(self, data, health_labels, contam_labels):
                self.data = data
                self.health_labels = health_labels
                self.contam_labels = contam_labels

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return (
                    self.data[idx].unsqueeze(0),  # add channel dim → (1, bands, H, W)
                    self.health_labels[idx],
                    self.contam_labels[idx],
                )

        train_ds = DummyDataset(train_data, train_health, train_contam)
        val_ds = DummyDataset(val_data, val_health, val_contam)

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
        scheduler  = CosineAnnealingWarmRestarts(optimizer, T_0=cfg["cosine_T0"], T_mult=cfg["cosine_T_mult"])
        scaler     = GradScaler()
        ce_criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

        # Resume from checkpoint if provided
        if resume_path:
            start_epoch, best_auc = load_checkpoint(resume_path, model, optimizer, scheduler, device)
            patience_counter = 0
            best_epoch = start_epoch - 1
        else:
            best_auc = -1.0
            patience_counter = 0
            patience = 10  # Early stopping patience
            best_epoch = 0
            start_epoch = 0

        for epoch in range(start_epoch, cfg["epochs"]):
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
                best_epoch = epoch
                patience_counter = 0
                
                # Save full checkpoint for resuming
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_auc': best_auc,
                    'cfg': cfg
                }
                torch.save(checkpoint, cfg["save_path"])
                print(f"  ↳ saved best model (epoch {epoch}, mean AUC {best_auc:.4f})")
            else:
                patience_counter += 1
                print(f"  ↳ no improvement for {patience_counter} epochs")
                
            # Early stopping
            if patience_counter >= 10:  # patience
                print(f"\nEarly stopping triggered after 10 epochs without improvement")
                print(f"Best model was from epoch {best_epoch} with AUC {best_auc:.4f}")
                break

        print(f"\nTraining complete. Best val AUC: {best_auc:.4f} (epoch {best_epoch})")
        print(f"Model saved to: {cfg['save_path']}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Hyperspectral Soil Classification Model")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, help="Number of epochs (overrides config)")
    parser.add_argument("--lr", type=float, help="Learning rate (overrides config)")
    parser.add_argument("--batch_size", type=int, help="Batch size (overrides config)")
    parser.add_argument("--save_path", type=str, help="Save path for model (overrides config)")
    parser.add_argument("--calibrate", action="store_true", help="Enable radiometric calibration")
    parser.add_argument("--calib_config", type=str, help="Path to calibration config file")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def update_config_from_args(cfg: dict, args) -> dict:
    """Update config with command line arguments"""
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.lr:
        cfg["lr"] = args.lr
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.save_path:
        cfg["save_path"] = args.save_path
    
    return cfg


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    
    # Load configuration
    if args.config:
        cfg = load_config(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        cfg = CFG.copy()
        print("Using default configuration")
    
    # Override config with command line arguments
    cfg = update_config_from_args(cfg, args)
    
    # Print configuration
    print("Training configuration:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")
    print()
    
    # Start training
    train(cfg, resume_path=args.resume)
