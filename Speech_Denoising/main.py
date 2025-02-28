from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from download import download
from spectrogram import audio_to_spectrogram, normalize_spectrogram


print("Загрузка данных...")
download()

clean_files = sorted(Path('datasets/train/clean').glob('*.wav'))
noisy_files = sorted(Path('datasets/train/noisy').glob('*.wav'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")


def center_crop(tensor, target_tensor):
    _, _, h, w = tensor.size()
    _, _, target_h, target_w = target_tensor.size()
    delta_h = h - target_h
    delta_w = w - target_w
    crop_top = delta_h // 2
    crop_left = delta_w // 2
    return tensor[:, :, crop_top:crop_top + target_h, crop_left:crop_left + target_w]


class AudioDenoising(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.bottleneck = self.conv_block(256, 512)
        self.dec1 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256 * 2, 128)
        self.dec3 = self.upconv_block(128 * 2, 64)
        self.dec4 = self.upconv_block(64 * 2, 32)
        self.final = nn.Conv2d(32 * 2, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        bn = self.bottleneck(e4)

        d1 = self.dec1(bn)
        if d1.size()[2:] != e4.size()[2:]:
            e4 = center_crop(e4, d1)
        d1 = torch.cat([d1, e4], dim=1)

        d2 = self.dec2(d1)
        if d2.size()[2:] != e3.size()[2:]:
            e3 = center_crop(e3, d2)
        d2 = torch.cat([d2, e3], dim=1)

        d3 = self.dec3(d2)
        if d3.size()[2:] != e2.size()[2:]:
            e2 = center_crop(e2, d3)
        d3 = torch.cat([d3, e2], dim=1)

        d4 = self.dec4(d3)
        if d4.size()[2:] != e1.size()[2:]:
            e1 = center_crop(e1, d4)
        d4 = torch.cat([d4, e1], dim=1)

        out = self.final(d4)

        out = torch.nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out


def collate_fn(batch):
    noisy_list, clean_list, orig_times, phase_list = zip(*batch)
    max_time = max(orig_times)
    padded_noisy = []
    padded_clean = []
    for noisy, clean in zip(noisy_list, clean_list):
        pad_amount = max_time - noisy.shape[2]
        noisy_padded = torch.nn.functional.pad(noisy, (0, pad_amount), mode='constant', value=0)
        clean_padded = torch.nn.functional.pad(clean, (0, pad_amount), mode='constant', value=0)
        padded_noisy.append(noisy_padded)
        padded_clean.append(clean_padded)
    noisy_batch = torch.stack(padded_noisy)
    clean_batch = torch.stack(padded_clean)
    return noisy_batch, clean_batch, orig_times, phase_list


def train(model, train_loader, val_loader, epochs=50):
    criterion = lambda pred, target: 0.7 * nn.MSELoss()(pred, target) + 0.3 * nn.L1Loss()(pred, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        i = 0
        for noisy, clean, _, _ in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            if i % 20 == 0:
                print(f'Batch = {i}, Loss : {loss.item()}')
            i += 1
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for noisy, clean, _, _ in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                outputs = model(noisy)
                loss = criterion(outputs, clean)
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Эпоха {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Построение графиков
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.title('График обучения: Train и Val Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


clean_dir = Path('datasets/train/clean')
noisy_dir = Path('datasets/train/noisy')
noisy_dir.mkdir(parents=True, exist_ok=True)
snr_db = 10


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, clean_dir, noisy_dir):
        self.clean_files = sorted(Path(clean_dir).glob('*.wav'))
        self.noisy_files = sorted(Path(noisy_dir).glob('*.wav'))

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_audio, _ = librosa.load(str(self.clean_files[idx]), sr=16000)
        noisy_audio, _ = librosa.load(str(self.noisy_files[idx]), sr=16000)
        clean_spec, clean_phase = audio_to_spectrogram(clean_audio)
        noisy_spec, noisy_phase = audio_to_spectrogram(noisy_audio)
        original_time = clean_spec.shape[1]  # Сохраняем длину
        clean_spec = np.log1p(clean_spec)  # Без нормализации
        noisy_spec = np.log1p(noisy_spec)
        return (
            torch.tensor(noisy_spec).unsqueeze(0).float(),
            torch.tensor(clean_spec).unsqueeze(0).float(),
            original_time,
            clean_phase
        )


def process_and_save(model, noisy_file_path, output_file_path, sr=16000, n_fft=2048, hop_length=256):
    noisy_audio, _ = librosa.load(noisy_file_path, sr=sr)

    stft = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    magnitude_tensor = torch.tensor(magnitude).unsqueeze(0).unsqueeze(0).float().to(device)

    model.eval()
    with torch.no_grad():
        predicted_magnitude = model(magnitude_tensor).cpu().squeeze(0).squeeze(0).numpy()

    predicted_stft = predicted_magnitude * (np.cos(phase) + 1j * np.sin(phase))
    predicted_audio = librosa.istft(predicted_stft, hop_length=hop_length)

    sf.write(output_file_path, predicted_audio, sr)


model = AudioDenoising().to(device)
train_loader = torch.utils.data.DataLoader(
    AudioDataset('datasets/train/clean', 'datasets/train/noisy'),
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)

val_dataset = AudioDataset('datasets/test/clean', 'datasets/test/noisy')

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn
)

train(model, train_loader, val_loader, epochs=10)

test_loader = torch.utils.data.DataLoader(
    AudioDataset('datasets/test/clean', 'datasets/test/noisy'),
    batch_size=6,
    shuffle=False,
    collate_fn=collate_fn
)


def denormalize_spectrogram(normalized, mean, std):
    return normalized * std + mean


def plot_spectrograms(model, test_file):
    noisy_audio, _ = librosa.load(test_file, sr=16000)
    noisy_spec, _ = audio_to_spectrogram(noisy_audio)
    noisy_spec = normalize_spectrogram(noisy_spec)

    noisy_tensor = torch.tensor(noisy_spec).unsqueeze(0).unsqueeze(0).float().to(device)

    model.eval()
    with torch.no_grad():
        predicted_spec = model(noisy_tensor).cpu().squeeze(0).squeeze(0).numpy()

    process_and_save(model, test_file, 'denoised.wav')

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    librosa.display.specshow(noisy_spec, sr=16000, hop_length=256, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Зашумленная спектрограмма')

    plt.subplot(1, 2, 2)
    librosa.display.specshow(predicted_spec, sr=16000, hop_length=256, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Предсказанная чистая спектрограмма')

    plt.tight_layout()
    plt.show()


test_file = 'datasets/test/noisy/p230_107.wav'
plot_spectrograms(model, test_file)
