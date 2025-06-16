import os
from PIL import Image
from tqdm import tqdm
import shutil

# === CONFIG ===
RESIZE_SIZE = (224, 224)
MAX_SAMPLES = 1000

# === INPUT ===
DRONE_IMG_RAW_DIR = "C:/Users/ASUS/Downloads/archive/Synthetic_Drone_Classification_Dataset/train/no_drone"
RAW_AUDIO_DIR = "C:/Users/ASUS/DroneAudioDataset/Binary_Drone_Audio/unknown"

# === OUTPUT ===
DRONE_IMG_OUT = "C:/VSCode/Multi_Modal_Drone_Detection/data/not_drone/image"
DRONE_AUDIO_OUT = "C:/VSCode/Multi_Modal_Drone_Detection/data/not_drone/audio"

# === SETUP ===
os.makedirs(DRONE_IMG_OUT, exist_ok=True)
os.makedirs(DRONE_AUDIO_OUT, exist_ok=True)

# === COLLECT INPUT FILES ===
image_files = sorted([
    f for f in os.listdir(DRONE_IMG_RAW_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])
audio_files = sorted([f for f in os.listdir(RAW_AUDIO_DIR) if f.endswith(".wav")])

assert len(image_files) > 0, "❌ No images found in drone folder!"
assert len(audio_files) > 0, "❌ No audio found in yes_drone folder!"

# === PROCESS DRONE DATA ===
count = 0
print("\n📦 Processing drone image + audio pairs...")
for img_name in tqdm(image_files):
    if count >= min(MAX_SAMPLES, len(audio_files)):
        break

    # === Image ===
    img_path = os.path.join(DRONE_IMG_RAW_DIR, img_name)
    img = Image.open(img_path).convert("RGB").resize(RESIZE_SIZE)
    out_img_name = f"not_drone_{count:04d}.jpg"
    img.save(os.path.join(DRONE_IMG_OUT, out_img_name))

    # === Audio ===
    src_audio = os.path.join(RAW_AUDIO_DIR, audio_files[count])
    dst_audio = os.path.join(DRONE_AUDIO_OUT, f"not_drone_{count:04d}.wav")
    shutil.copy(src_audio, dst_audio)

    count += 1

print(f"\n✅ Saved {count} paired drone images and audio files!")
