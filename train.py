import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from model import SoilHSI3DCNN
from dataset import HyperspectralSoilDataset
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

# Example usage – replace with your paths
data_paths = [...]   # list of .npy cube paths
labels = [...]       # list of (health_class, [metal, pfas, glyphosate, mp] probs)

train_paths, val_paths, train_lbl, val_lbl = train_test_split(data_paths, labels, test_size=0.2, random_state=42)

train_ds = HyperspectralSoilDataset(train_paths, train_lbl, train=True)
val_ds   = HyperspectralSoilDataset(val_paths,   val_lbl,   train=False)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

model = SoilHSI3DCNN(num_bands=200, num_classes=5, num_contaminants=4).cuda()
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
scaler = GradScaler()
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

for epoch in range(100):
    model.train()
    for cube, health, contam in tqdm(train_loader):
        cube, health, contam = cube.cuda(), health.cuda(), contam.cuda()
        optimizer.zero_grad()

        with autocast():
            pred_health, pred_contam = model(cube)
            loss = criterion(pred_health, health) + F.binary_cross_entropy(pred_contam, contam)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()
    # validation + ROC-AUC logging here...
    print(f"Epoch {epoch} completed – loss: {loss.item():.4f}")

torch.save(model.state_dict(), "soil_3dcnn_enotrium.pth")
