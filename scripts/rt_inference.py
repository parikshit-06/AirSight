import cv2
import torch
import numpy as np
import sounddevice as sd
import torchaudio
import time

# ========== Load Model ==========
class MultiModalModel(torch.nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()
        self.img_branch = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten()
        )
        self.audio_branch = torch.nn.Sequential(
            torch.nn.Linear(40, 64),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64 + 16, 2),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, img, mfcc):
        img_feat = self.img_branch(img)
        audio_feat = self.audio_branch(mfcc)
        combined = torch.cat((img_feat, audio_feat), dim=1)
        return self.classifier(combined)

model = MultiModalModel()
model.load_state_dict(torch.load('model/fusion_model.pth', map_location=torch.device('cpu')))
model.eval()

# ========== Preprocessing Functions ==========
def preprocess_image(frame):
    frame = cv2.resize(frame, (64, 64))
    frame = frame / 255.0
    tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return tensor

def extract_mfcc(audio, fs=16000):
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    mfcc = torchaudio.transforms.MFCC(sample_rate=fs, n_mfcc=40)(waveform)
    mfcc = mfcc.mean(dim=2)  # (1, 40)
    return mfcc

def get_audio_sample(duration=1, fs=16000):
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

# ========== Real-time Loop (inference every 1s) ==========
cap = cv2.VideoCapture(0)
fs = 16000
last_inference_time = time.time()
label_text = "Waiting..."

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if current_time - last_inference_time >= 1.0:
            audio = get_audio_sample(duration=1, fs=fs)
            img_tensor = preprocess_image(frame)
            mfcc_tensor = extract_mfcc(audio, fs=fs)

            with torch.no_grad():
                pred = model(img_tensor, mfcc_tensor)
                label = torch.argmax(pred, dim=1).item()
                label_text = "Drone" if label == 1 else "No Drone"

            last_inference_time = current_time

        # Show latest prediction continuously
        cv2.putText(frame, f"Prediction: {label_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Drone Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted")

cap.release()
cv2.destroyAllWindows()