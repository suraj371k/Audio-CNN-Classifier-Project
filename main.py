import base64
import io
import modal
import numpy as np
import requests
import torch.nn as nn
import torchaudio.transforms as T
import torch
from pydantic import BaseModel
import soundfile as sf
import librosa

from model import simpleResNetAudioCNN

app = modal.App("audio-cnn-classifier")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         # used to read audio files
         .apt_install(["libsndfile1"])
         .add_local_python_source("model"))

model_volume = modal.Volume.from_name("esc-model")


class AudioProcessor:
    def __init__(self):
        self.transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=44100,
                n_fft=1024,
                hop_length=512,
                n_mels=128,
                f_min=0,
                f_max=11025
            ),
            T.AmplitudeToDB()
        )

    def process_audio_chunk(self, audio_data):
        waveform = torch.from_numpy(audio_data).float()

        waveform = waveform.unsqueeze(0)

        # add another channel dimension to the spectrogram
        spectrogram = self.transform(waveform)

        return spectrogram.unsqueeze(0)

# Pydantics model for request validation


class InferenceRequest(BaseModel):
    audio_data: str


@app.cls(image=image, gpu="A10G", volumes={"/models": model_volume}, scaledown_window=15)
class AudioClassifier:
    @modal.enter()
    def load_model(self):
        print("Loading models on enter")
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load('/models/best_model.pth',
                                map_location=self.device)
        self.classes = checkpoint['classes']

        self.model = simpleResNetAudioCNN(num_classes=len(self.classes))
        # load weights into model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.audio_processor = AudioProcessor()
        print("Model loaded on enter")

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):
        # decoding base64 audio data
        audio_bytes = base64.b64decode(request.audio_data)
        print(f"Decoded bytes length: {len(audio_bytes)}")    # Should be >0

        # read audio data from bytes
        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.seek(0)  # Reset buffer position!
        audio_data, sample_rate = sf.read(audio_buffer, dtype="float32")
        # Key check
        print(
            f"sf.read: audio_data shape {audio_data.shape}, sr {sample_rate}")

        # if stereo, convert to mono by averaging channels
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # resample to 44100 Hz if necessary
        if sample_rate != 44100:
            audio_data = librosa.resample(
                y=audio_data, orig_sr=sample_rate, target_sr=44100)

        spectrogram = self.audio_processor.process_audio_chunk(audio_data)
        spectrogram = spectrogram.to(self.device)

        with torch.no_grad():
            output, feature_maps = self.model(
                spectrogram, return_feature_maps=True)

            # Handle NaN values in output, means replaces NaNs with zeros
            output = torch.nan_to_num(output)

            # Apply softmax to get probabilities, softmax makesure all values are between 0 and 1 and sum to 1
            # here, dim = 0 is batch dim, dim = 1 is class dim (batch_size, num_classes)
            probabilities = torch.softmax(output, dim=1)

            # Get top 3 predictions
            top3_probs, top3_indicies = torch.topk(probabilities[0], 3)

            # Prepare predictions, example format: top3_probs: [0.9, 0.05, 0.03], top3_indicies: [15, 42, 8]
            # (0.9, 15), (0.05, 42), (0.03, 8)
            predictions = [{"class": self.classes[idx.item()], "confidence": prob.item()}
                           for prob, idx in zip(top3_probs, top3_indicies)]

            # Prepare visualization data from feature maps
            viz_data = {}
            for name, tensor in feature_maps.items():
                if tensor.dim() == 4:  # [batch_size, channels, height, width]
                    aggregated_tensor = torch.mean(tensor, dim=1)
                    squeezed_tensor = aggregated_tensor.squeeze(
                        0)  # remove batch dim
                    # convert to numpy instead of tensor
                    numpy_array = squeezed_tensor.cpu().numpy()
                    clean_array = np.nan_to_num(numpy_array)
                    viz_data[name] = {
                        "shape": list(clean_array.shape),
                        "values": clean_array.tolist()
                    }
            # Prepare clean spectrogram for response
            # we do it twice -> [batch_size, channels, height, width] -> [height, width]
            spectrogram_np = spectrogram.squeeze(0).squeeze(0).cpu().numpy()
            # we will convert NaN to number (0) for JSON serialization
            clean_spectrogram = np.nan_to_num(spectrogram_np)

            max_samples = 8000
            waveform_sample_rate = 44100
            if len(audio_data) > max_samples:
                step = len(audio_data) // max_samples
                # downsample and creates a new array
                waveform_data = audio_data[::step]
            else:
                waveform_data = audio_data

        response = {
            "predictions": predictions,
            "visualization": viz_data,
            "input_spectrogram": {
                "shape": list(clean_spectrogram.shape),
                "values": clean_spectrogram.tolist()
            },
            "waveform": {
                "values": waveform_data.tolist(),
                "sample_rate": 44100,
                "duration": len(audio_data) / 44100
            }
        }

        return response


@app.local_entrypoint()
def main():
    audio_data, sample_rate = sf.read("4th.wav")
    # Debug here too
    print(f"Local file: shape {audio_data.shape}, sr {sample_rate}")

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    payload = {"audio_data": audio_b64}

    server = AudioClassifier()
    url = server.inference.get_web_url()
    response = requests.post(url, json=payload)
    response.raise_for_status()

    result = response.json()

    waveform_info = result.get("waveform", {})
    if waveform_info:
        values = waveform_info.get("values", {})
        print(f"First 10 values: {[round(v, 4) for v in values[:10]]}...")
        print(f"Duration: {waveform_info.get("duration", 0)}")

    print("Top predictions:")
    for pred in result.get("predictions", []):
        print(f"  -{pred["class"]} {pred["confidence"]:0.2%}")
