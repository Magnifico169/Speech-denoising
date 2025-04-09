import datetime
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pesq import pesq
from pystoi import stoi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

SR = 16000
N_FFT = 1024
HOP_LENGTH = 256


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, clean_directory, noisy_directory, global_mean_=None, global_std_=None):
        self.clean_files = sorted(Path(clean_directory).glob('*.wav'))
        self.noisy_files = sorted(Path(noisy_directory).glob('*.wav'))
        self.global_mean = global_mean_
        self.global_std = global_std_

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_audio, _ = librosa.load(self.clean_files[idx], sr=SR)
        noisy_audio, _ = librosa.load(self.noisy_files[idx], sr=SR)
        clean_spec = audio_to_spectrogram(clean_audio)
        noisy_spec = audio_to_spectrogram(noisy_audio)
        if self.global_mean is not None and self.global_std is not None:
            clean_spec = (clean_spec - self.global_mean) / self.global_std
            noisy_spec = (noisy_spec - self.global_mean) / self.global_std
        return (
            torch.tensor(noisy_spec).unsqueeze(0).float(),
            torch.tensor(clean_spec).unsqueeze(0).float(),
            clean_spec.shape[1],
            None
        )


def audio_to_spectrogram(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_magnitude = np.log1p(magnitude)
    return log_magnitude


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
        self.enc1 = self.encode_block(1, 16)
        self.enc2 = self.encode_block(16, 32)
        self.enc3 = self.encode_block(32, 64)
        self.enc4 = self.encode_block(64, 128)
        self.bottleneck = self.bottle_neck(128, 256)
        self.dec1 = self.decode_block(256, 128)
        self.dec2 = self.decode_block(128 * 2, 64)
        self.dec3 = self.decode_block(64 * 2, 32)
        self.dec4 = self.decode_block(32 * 2, 16)
        self.final = nn.Conv2d(16 * 2, 1, kernel_size=1)

    @staticmethod
    def bottle_neck(in_channels, out_channels, dropout=0.2):
        return nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def encode_block(in_channels, out_channels, dropout=0.2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2)
        )

    @staticmethod
    def decode_block(in_channels, out_channels, dropout=0.2):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
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


def train(model: AudioDenoising, train_load, val_load, epochs=50):
    print(f"Начало обучения, текущее время: {datetime.datetime.now()}")
    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for i, (noisy, clean, _, _) in enumerate(train_load):
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
            if i % 20 == 0:
                print(f"Батч {i}, Потеря: {loss.item()}")
        avg_train_loss = epoch_train_loss / len(train_load)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for noisy, clean, _, _ in val_load:
                noisy, clean = noisy.to(device), clean.to(device)
                outputs = model(noisy)
                loss = criterion(outputs, clean)
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / len(val_load)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_train_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Ранняя остановка активирована")
                break

        print(f"Эпоха {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.title('График обучения')
    plt.legend()
    plt.grid(True)
    plt.show()


def process_and_save(model: AudioDenoising, noisy_file_path, output_file_path, sr=SR, hop_length=HOP_LENGTH):
    noisy_audio, _ = librosa.load(noisy_file_path, sr=sr)
    noisy_spec, noisy_phase = audio_to_spectrogram(noisy_audio)
    noisy_tensor = torch.tensor(noisy_spec).unsqueeze(0).unsqueeze(0).float().to(device)

    model.eval()
    with torch.no_grad():
        predicted_magnitude = model(noisy_tensor).cpu().squeeze(0).squeeze(0).numpy()
        pred_magnitude = np.expm1(predicted_magnitude)
    pred_stft = pred_magnitude * np.exp(1j * noisy_phase)
    predicted_audio = librosa.istft(pred_stft, hop_length=hop_length)
    sf.write(output_file_path, predicted_audio, sr)


def evaluate_model(model: AudioDenoising, test_load, mean: float, std: float, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
    model.eval()
    pesq_scores, stoi_scores = [], []
    device_ = next(model.parameters()).device

    for batch_idx, (noisy, clean, orig_times, _) in enumerate(test_load):
        noisy = noisy.to(device_)
        pred_spec = model(noisy)
        pred_spec = pred_spec * std + mean
        pred_spec = torch.expm1(pred_spec).detach().cpu().numpy()

        for i in range(noisy.size(0)):
            dataset_idx = batch_idx * test_load.batch_size + i
            if dataset_idx >= len(test_load.dataset):
                break
            clean_file = test_load.dataset.clean_files[dataset_idx]
            noisy_file = test_load.dataset.noisy_files[dataset_idx]
            clean_audio, _ = librosa.load(clean_file, sr=sr)
            noisy_audio, _ = librosa.load(noisy_file, sr=sr)
            stft_noisy = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length)
            noisy_phase = np.angle(stft_noisy)
            pred_mag = pred_spec[i].squeeze()

            min_freq = min(pred_mag.shape[0], noisy_phase.shape[0])
            min_time = min(pred_mag.shape[1], noisy_phase.shape[1])
            pred_mag = pred_mag[:min_freq, :min_time]
            noisy_phase = noisy_phase[:min_freq, :min_time]
            pred_stft = pred_mag * np.exp(1j * noisy_phase)
            pred_audio = librosa.istft(pred_stft, hop_length=hop_length)
            min_length = min(len(clean_audio), len(pred_audio))
            clean_clipped = clean_audio[:min_length]
            pred_clipped = pred_audio[:min_length]
            if min_length == 0:
                print(f"Пропуск короткого аудио: {clean_file}")
                continue
            try:
                pesq_score = pesq(sr, clean_clipped, pred_clipped, 'wb')
                pesq_scores.append(pesq_score)
            except Exception as e:
                print(f"Ошибка PESQ в файле {clean_file}: {e}")
            try:
                stoi_score = stoi(clean_clipped, pred_clipped, sr)
                stoi_scores.append(stoi_score)
            except Exception as e:
                print(f"Ошибка STOI в файле {clean_file}: {e}")

    print(f"Средний PESQ: {np.mean(pesq_scores):.3f}" if pesq_scores else "PESQ не рассчитан")
    print(f"Средний STOI: {np.mean(stoi_scores):.3f}" if stoi_scores else "STOI не рассчитан")


def compute_global_mean_std(dataset):
    sum_val = 0.0
    sum_sq_val = 0.0
    total_pixels = 0
    for idx in range(len(dataset)):
        noisy, clean, _, _ = dataset[idx]
        noisy_np = noisy.numpy()
        clean_np = clean.numpy()
        combined = np.concatenate([noisy_np, clean_np], axis=0)
        sum_val += combined.sum()
        sum_sq_val += np.square(combined).sum()
        total_pixels += combined.size
    global_mean = sum_val / total_pixels
    global_std = np.sqrt((sum_sq_val / total_pixels) - (global_mean ** 2))
    return global_mean, global_std


train_dataset = AudioDataset('datasets/train/clean', 'datasets/train/noisy')


def global_items(train_dataset_):
    file_path = 'for_global_items.txt'
    if not Path(file_path).exists() or Path(file_path).stat().st_size == 0:
        global_mean, global_std = compute_global_mean_std(train_dataset_)
        with open(file_path, 'w') as file:
            file.write(f'{global_mean}\n{global_std}')
    else:
        with open(file_path, 'r') as file:
            list_text = file.read().split('\n')
            global_mean = float(list_text[0])
            global_std = float(list_text[1])
    return global_mean, global_std


global_mean, global_std = global_items(train_dataset)


train_loader = DataLoader(
    AudioDataset('datasets/train/clean', 'datasets/train/noisy', global_mean, global_std),
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    AudioDataset('datasets/valid/clean', 'datasets/valid/noisy', global_mean, global_std),
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    AudioDataset('datasets/test/clean', 'datasets/test/noisy', global_mean, global_std),
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)


def main():
    model = AudioDenoising().to(device)
    train(model, train_loader, val_loader, epochs=1)
    torch.save(model.state_dict(), 'speech_denoising.pth')
    evaluate_model(model, test_loader, global_mean, global_std)


if __name__ == '__main__':
    main()
