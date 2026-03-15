import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from huggingface_hub import hf_hub_download

# ── Config ──
IMG_SIZE    = 300
DEVICE      = torch.device('cpu')  # Streamlit runs on CPU
HF_REPO     = 'Salonideshmukh/dr-detection-model'
MODEL_FILE  = 'best_dr_model.pth'

# ── Model definition (exact same as training) ──
class DRClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=False, num_classes=0, global_pool=''
        )
        feat_dim = self.backbone.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        self.gradients  = None
        self.activations = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        feats = self.backbone(x)
        if feats.requires_grad:
            feats.register_hook(self.save_gradient)
        self.activations = feats
        return self.head(self.pool(feats))


@torch.no_grad()
def load_model():
    """Download model from HuggingFace and load weights."""
    model_path = hf_hub_download(repo_id=HF_REPO, filename=MODEL_FILE)
    model      = DRClassifier(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


# ── Preprocessing (identical to training) ──
def crop_black_borders(img, threshold=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return img[y:y+h, x:x+w]

def ben_graham_normalization(img, sigmaX=10):
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def preprocess_image(img_array):
    """
    Input  : numpy array in RGB (from Streamlit uploader)
    Output : preprocessed numpy array in RGB
    This is the key fix — we convert RGB→BGR for OpenCV processing,
    then convert back to RGB for display. Same result as training.
    """
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # RGB → BGR for OpenCV
    img = crop_black_borders(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = apply_clahe(img)
    img = ben_graham_normalization(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)         # BGR → RGB for display
    return img

def to_tensor(img_rgb):
    """Normalize and convert to tensor — identical to val_transform in training."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = img_rgb.astype(np.float32) / 255.0
    img  = (img - mean) / std
    img  = torch.from_numpy(img).permute(2, 0, 1).float()  # (3, H, W)
    return img


def predict(model, img_array, threshold=0.35):
    """
    Full inference pipeline.
    img_array : numpy RGB array from Streamlit
    Returns   : preprocessed image, grad-cam, prediction, probability
    """
    # Preprocess
    img_rgb    = preprocess_image(img_array)
    img_tensor = to_tensor(img_rgb)

    # ── Grad-CAM ──
    model.eval()
    model.backbone.set_grad_checkpointing(enable=False)

    inp = img_tensor.unsqueeze(0).to(DEVICE)

    features = model.backbone(inp)
    features.retain_grad()
    pooled   = model.pool(features)
    output   = model.head(pooled)

    pred_class = output.argmax(dim=1).item()
    prob       = F.softmax(output, dim=1)[0, 1].item()

    model.zero_grad()
    output[0, pred_class].backward()

    grads   = features.grad
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam     = F.relu((weights * features).sum(dim=1, keepdim=True))
    cam     = F.interpolate(cam, (IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    cam     = cam.squeeze().detach().numpy()
    cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    prediction = 'DR' if prob >= threshold else 'No DR'
    return img_rgb, cam, prediction, prob