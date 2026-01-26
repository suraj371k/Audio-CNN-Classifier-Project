from csv import writer
from pathlib import Path
import sys
from typing import Self
import pandas as pd
import modal

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from model import simpleResNetAudioCNN

app = modal.App("audio-cnn-classifier")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
         .run_commands([
             "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
             "cd /tmp && unzip esc50.zip",
             "mkdir -p /opt/esc50-data",
             "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
             "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
         ])
         .add_local_python_source("model")
         )

volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)

# to load and get the dataset in quick manner


class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform
        # filter the training and validation data

        if split == "train":
            self.metadata = self.metadata[self.metadata["fold"] != 5]
        else:
            self.metadata = self.metadata[self.metadata["fold"] == 5]

        # sorting the unique classes only
        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(
            self.classes)}   # encoding in short
        self.metadata['label'] = self.metadata['category'].map(
            self.class_to_idx)  # mapping the label and creating new column

    def __len__(self):
        return len(self.metadata)

    # will create a function to extract and sample from the dataset
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row['filename']

        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform  # if no transform is applied

        return spectrogram, row['label']


def mixup_data(x, y):
    lam = np.random.beta(0.2, 0.2)   # x -> features, y -> labels

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # (o.7 * audio1) + (0.3 * audio2)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# used to claculate the loss


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + 1 - lam * criterion(pred, y_b)


@app.function(image=image, gpu="A10G", volumes={"/data": volume, "/models": model_volume}, timeout=60 * 60 * 3)
def train():
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'/models/tensorboard_logs/run_{timestamp}'
    writer = SummaryWriter(log_dir)

    esc50_dir = Path("/opt/esc50-data")

    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        # convert to decibles
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30),
        # these two are like dropouts(to prevent overfitting) but for spectrograms/audio
        T.TimeMasking(time_mask_param=80)
    )

    validation_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB()
    )

    train_dataset = ESC50Dataset(
        data_dir=esc50_dir, metadata_file=esc50_dir / "meta" / "esc50.csv", split="train", transform=train_transform)

    validation_dataset = ESC50Dataset(
        data_dir=esc50_dir, metadata_file=esc50_dir / "meta" / "esc50.csv", split="validation", transform=validation_transform)

    print("Training samples: " + str(len(train_dataset)))
    print("Validation samples: " + str(len(validation_dataset)))

    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(
        validation_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = simpleResNetAudioCNN(num_classes=len(train_dataset.classes))
    model.to(device)

    num_epochs = 100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # optimizers job is to tune the weights and biases of the entire model
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    # scheduler adjusts the learning rate during training so that model converges better
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1
    )

    best_accuracy = 0.0

    print("Starting to train the model...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)

            if np.random.random() > 0.7:
                # apply the mixup
                data, target_a, target_b, lam = mixup_data(data, target)
                output = model(data)
                loss = mixup_criterion(
                    criterion, output, target_a, target_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item(): .4f}'})

        avg_epoch_loss = epoch_loss / len(train_dataloader)

        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
        writer.add_scalar(
            'Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # validation after each epoch
        model.eval()

        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(test_dataloader)

        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)

        print(
            f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch,
                'classes': train_dataset.classes
            }, '/models/best_model.pth')
            print(f'New best model saved: {accuracy:.2f}%')

    writer.close()
    print(f'Training completed! Best accuracy: {best_accuracy:.2f}%')


@app.local_entrypoint()
def main():
    train.remote()
