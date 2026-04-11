# Dockerfile - Hyperspectral-Restruct (Production + Edge)
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps + PyTorch (for ONNX export)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source and export model to ONNX (one-time during build)
COPY . .
RUN python -c "
import torch
from model import Hyperspectral3DCNN
model = Hyperspectral3DCNN(num_classes=1)
model.load_state_dict(torch.load('checkpoints/best_nitrogen_model.pth', map_location='cpu', weights_only=True))
model.eval()
dummy_input = torch.randn(1, 1, 200, 64, 64)  # (batch, channels, bands, H, W) - adjust to your cube shape
torch.onnx.export(model, dummy_input, 'models/nitrogen_model.onnx',
                  export_params=True, opset_version=17,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
print('✅ Model exported to ONNX for edge inference')
"

# === Runtime stage (lightweight for edge / cloud) ===
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt onnxruntime fastapi uvicorn pillow rasterio shapely

# Copy runtime code + exported ONNX model
COPY --from=builder /app/api.py /app/models/nitrogen_model.onnx /app/models/
COPY --from=builder /app/dataset.py /app/model.py /app/ /app/configs/ /app/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
