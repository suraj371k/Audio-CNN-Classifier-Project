# Audio CNN Classifier & Visualizer 🧠🎵

[![Model Accuracy](https://img.shields.io/badge/Model%20Accuracy-83.4%25-brightgreen.svg)]()
[![Next.js](https://img.shields.io/badge/Next.js-14-blue.svg)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://reactjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![Modal](https://img.shields.io/badge/Modal-Serverless-red.svg)](https://modal.com/)

**Production-ready audio-CNN classification** trained on ResNet architecture which i created from scratch, then trained the model on **ESC-50 DATASET** achieving **83.4% accuracy**. Live **feature map visualization**, **ResNet-50**, **Modal serverless**, **TensorBoard**, **Next.js 14 + React**.

---

## ✨ Features

- **83.4% Top-1 Accuracy** on ESC-50 (50 environmental sounds)
- **Live Feature Maps** - Watch ResNet-50 layers activate
- **Real-time Waveform** + Mel Spectrogram visualization
- **50-class Emoji Support** 🐦👏🚁🔨 (chirping_birds, clapping, etc.)
- **Modal A10G GPU** serverless inference (~45ms)
- **Next.js 14 App Router** + TypeScript + shadcn/ui
- **Base64 WAV** upload → instant predictions
---
## 📊 Performance

| Metric | Value |
|--------|-------|
| **Top-1 Accuracy** | **83.4%** |
| Inference Time | **45ms** (A10G) |
| Classes | 50 |
| Input Length | 5s WAV (44.1kHz) |
| Spectrogram | 128 mel bins |

---

**Sample Output (clapping.wav):**
👏 clapping: 81.63%
👣 footsteps: 18.36%
🥫 can_opening: 0.01%

---

## 🏗️ Architecture

**ResNet-50 for Audio Spectrograms** (3-4-6-3 blocks):

`Conv1(7×7,64) → Layer1(3×64ch) → Layer2(4×128ch) → Layer3(6×256ch) → Layer4(3×512ch)`
↓
`AdaptiveAvgPool2d → Dropout(0.5) → FC(512→50) → Softmax`

--- 
**Audio Pipeline:**
`WAV → Mono → Resample(44.1kHz) → MelSpec(n_mels=128,n_fft=1024,hop=512) → dB → ResNet`

---

## 🚀 Quick Start

### Backend (Modal)
```bash
pip install -r requirements.txt
modal token set
modal volume put esc-model best_model.pth
modal deploy main.py
```

### Frontend
```bash
cd frontend
pnpm install
pnpm dev
```

Live API: https://harshit7271--audio-cnn-classifier-audioclassifier-inference.modal.run/

---
## Structure
├── frontend/                 # Next.js 14 T3 Stack

│   ├── app/                 # App Router

│   ├── components/ui/       # shadcn/ui (Badge, Card, Progress)

│   └── lib/utils.ts        # API helpers

├── backend/                 # Modal + PyTorch

│   ├── main.py/               # FastAPI + AudioClassifier

|   ├── train.py/             

│   ├── model.py/            # ResNet-50 (16 blocks)

│   ├── requirements.txt/

│   └── best_model.pth/      # 83.4% checkpoint

└── README.md

---

# Code Highligths
### Backend(main.py)
```python
# Base64 → WAV → MelSpec → ResNet → Feature Maps
audio_bytes = base64.b64decode(request.audio_data)
spectrogram = MelSpectrogram(sample_rate=44100, n_mels=128)
output, feature_maps = model(spectrogram, return_feature_maps=True)
```

## Frontend(React)
```tsx
const base64String = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
const { predictions, visualization, waveform } = await response.json();
```
---

# Key Features
- GPU Autoscaling - Modal scales to zero

- NaN Handling - `torch.nan_to_num()` for robust inference

- Stereo→Mono - `np.mean(audio_data, axis=1)`

- Waveform Downsampling - Max 8000 samples for viz

- Layer Splitting - `splitLayers()` for block visualization

- ESC-50 Emojis - 50-class mapping (🐕🌧️👶🚪 etc.)

---
# Training
```bash
Dataset: ESC-50 (2000 clips, 5s, 44.1kHz)
Splits: 5-fold CV (Fold 5 = test)
Best: Epoch 100, Val Acc: 83.4%
```
---
#  UI Components

- Top Predictions
- Input Spectrogram
- Audio Waveform
- Convolutional Layer Outputs


---
# License
MIT - Use freely!

