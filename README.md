# AirSight
AirSight is a real-time multimodal drone detection system that uses both camera and microphone data. It combines CNN-based visual features with MFCC audio features in a lightweight neural network, making it suitable for deployment on edge devices like Jetson Nano or Raspberry Pi.

---

## Overview

AirSight leverages two sensory modalities—image and sound—for robust drone classification. It is built on PyTorch and OpenCV and supports live webcam feed and microphone input for continuous detection.

### Key Features
- **Visual Detection**: Captures and processes 64x64 RGB frames from a webcam.
- **Audio Detection**: Captures 1-second microphone samples and extracts MFCCs.
- **Multimodal Fusion**: Combines visual and audio features in a neural network.
- **Real-Time Inference**: Fast enough for real-time classification.
- **Lightweight Model**: Designed to run on laptops and edge devices.

---

## Project Structure

```
AirSight/
│
├── model/
│   └── fusion_model.pth           # Trained model weights
│
├── scripts/
│   └── rt_inference.py            # Real-time inference script
│
├── data/                          # Paired image/audio dataset
│
├── LICENSE                        # MIT Licensed
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/airsight.git
   cd airsight
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have a microphone and webcam connected.

---

## Running Real-Time Inference

```bash
python scripts/rt_inference.py
```

Press `q` to quit the live window.

---

## Model Architecture

```text
Image Input  → CNN (Conv → ReLU → Pool → Flatten)
Audio Input  → MFCC → Linear → ReLU
              ↓
          Concatenate → Linear → Softmax
```

---

## Training (Optional)

Training scripts are not included but the dataset was created by pairing:

- **Images** from: [Drone Type Classification (Kaggle)](https://www.kaggle.com/datasets/balajikartheek/drone-type-classification)
- **Audio** from: [Drone Audio Dataset (GitHub)](https://github.com/saraalemadi/DroneAudioDataset)

Each image-audio pair was labeled as "Drone" or "No Drone".

---

## Applications

- Perimeter surveillance
- Anti-drone defense

---

## Disclaimer

This project is intended for research and educational purposes. Accuracy and real-world robustness may vary with conditions.

---

## License

[MIT License](LICENSE)

---
Developed by Parikshit Sonwane, IIT Madras  
Contributions are welcome!
