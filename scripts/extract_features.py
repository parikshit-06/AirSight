import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import librosa
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2

# === CONFIG ===
IMG_SIZE = 224
MFCC_DIM = 40
AUDIO_DURATION = 2.5  # seconds
SR = 22050
ROOT_DATA = "data"
SAVE_PATH = "data/processed"
os.makedirs(SAVE_PATH, exist_ok=True)

# === TRANSFORM FOR IMAGE ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === LOAD MODEL FOR IMAGE FEATURES ===
model = mobilenet_v2(pretrained=True)
model.classifier = torch.nn.Identity()  # remove final classifier
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def extract_image_feature(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(image)
        return feat.squeeze().cpu().numpy()
    except Exception as e:
        print(f"[Image Error] {img_path}: {e}")
        return None

def extract_audio_feature(audio_path):
    try:
        y, _ = librosa.load(audio_path, sr=SR, duration=AUDIO_DURATION)
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=MFCC_DIM)
        return mfcc.mean(axis=1)
    except Exception as e:
        print(f"[Audio Error] {audio_path}: {e}")
        return None

def process_folder(label_name, label_val):
    img_dir = os.path.join(ROOT_DATA, label_name, "image")
    audio_dir = os.path.join(ROOT_DATA, label_name, "audio")
    paired = sorted(os.listdir(img_dir))

    X_img, X_audio, y = [], [], []

    for fname in tqdm(paired, desc=f"Processing {label_name}"):
        name = os.path.splitext(fname)[0]
        img_path = os.path.join(img_dir, f"{name}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, f"{name}.png")

        audio_path = os.path.join(audio_dir, f"{name}.wav")
        if not (os.path.exists(img_path) and os.path.exists(audio_path)):
            continue

        img_feat = extract_image_feature(img_path)
        aud_feat = extract_audio_feature(audio_path)

        if img_feat is not None and aud_feat is not None:
            X_img.append(img_feat)
            X_audio.append(aud_feat)
            y.append(label_val)

    return X_img, X_audio, y

# === MAIN ===
X_img1, X_aud1, y1 = process_folder("drone", 1)
X_img0, X_aud0, y0 = process_folder("not_drone", 0)

X_img = np.array(X_img1 + X_img0)
X_aud = np.array(X_aud1 + X_aud0)
y = np.array(y1 + y0)

np.save(os.path.join(SAVE_PATH, "X_image.npy"), X_img)
np.save(os.path.join(SAVE_PATH, "X_audio.npy"), X_aud)
np.save(os.path.join(SAVE_PATH, "y.npy"), y)

print(f"✅ Saved features: {X_img.shape=}, {X_aud.shape=}, {y.shape=}")