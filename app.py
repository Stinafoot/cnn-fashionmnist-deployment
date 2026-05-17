import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

# ── FashionMNIST labels ───────────────────────────────────────────────────────
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# ── Model (must match train.py exactly) ──────────────────────────────────────
class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Load model at startup ─────────────────────────────────────────────────────
device = torch.device("cpu")
model  = FashionCNN()
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

# ── Preprocessing pipeline ────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Grayscale(),                           # ensure single channel
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="FashionMNIST Prediction API",
    description="Upload a 28×28 greyscale image and receive a clothing category prediction.",
    version="1.0.0",
)

# ── Response schemas ──────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str

class PredictionResponse(BaseModel):
    predicted_class: str
    class_index: int
    confidence: float
    all_probabilities: dict[str, float]


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health():
    """Liveness check — returns ok if the service is running and the model is loaded."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(..., description="PNG/JPG image to classify")):
    """
    Upload a greyscale clothing image.
    Returns the predicted class, confidence score, and full probability distribution.
    """
    # ── Validate content type ─────────────────────────────────────────────────
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Send image/png or image/jpeg.",
        )

    # ── Read & preprocess ─────────────────────────────────────────────────────
    try:
        raw   = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("L")   # force greyscale
        tensor = preprocess(image).unsqueeze(0)            # (1, 1, 28, 28)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze()    # (10,)

    class_idx  = int(probs.argmax())
    confidence = float(probs[class_idx])
    all_probs  = {CLASSES[i]: round(float(probs[i]), 6) for i in range(len(CLASSES))}

    return PredictionResponse(
        predicted_class=CLASSES[class_idx],
        class_index=class_idx,
        confidence=round(confidence, 6),
        all_probabilities=all_probs,
    )
