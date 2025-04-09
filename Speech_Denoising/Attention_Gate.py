from pathlib import Path
import librosa
import torch
import numpy as np
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")


def process_tensors(tensor_a: torch.Tensor, tensor_b: torch.Tensor):
    if tensor_a.shape[-1] < tensor_b.shape[-1]:
        tensor_b = tensor_b[:, :, :, :tensor_a.shape[-1]]
    elif tensor_a.shape[-1] > tensor_b.shape[-1]:
        tensor_a = tensor_a[:, :, :, :tensor_b.shape[-1]]
    if tensor_a.shape[-2] < tensor_b.shape[-2]:
        tensor_b = tensor_b[:, :, :tensor_a.shape[-2], :]
    elif tensor_a.shape[-2] > tensor_b.shape[-2]:
        tensor_a = tensor_a[:, :, :tensor_b.shape[-2], :]
    return tensor_a, tensor_b


class Attention_Block(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.query = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, kernel_size=1),
            torch.nn.BatchNorm2d(output_channels)
        )
        self.key = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, kernel_size=1),
            torch.nn.BatchNorm2d(output_channels)
        )
        self.attention = torch.nn.Sequential(
            torch.nn.Conv2d(output_channels, output_channels, kernel_size=1),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.Sigmoid()
        )

    def forward(self, decoderr_input, encoderr_input):
        query = self.query(decoderr_input)
        key = self.key(encoderr_input)
        if query.shape[-1] < key.shape[-1]:
            key = key[:, :, :, :query.shape[-1]]
            encoderr_input = encoderr_input[..., :query.shape[-1]]
        elif query.shape[-1] > key.shape[-1]:
            query = query[:, :, :, :key.shape[-1]]
        if query.shape[-2] < key.shape[-2]:
            key = key[:, :, :query.shape[-2], :]
            encoderr_input = encoderr_input[:, :, :query.shape[-2], :]
        elif query.shape[-2] > key.shape[-2]:
            query = query[:, :, :key.shape[-2], :]
        attention_mask = self.attention(torch.relu(query + key))
        return encoderr_input * attention_mask


class DenoisingUnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_1 = self.encoder_block(input_channels=1, output_channels=16)
        self.encoder_2 = self.encoder_block(input_channels=16, output_channels=32)
        self.encoder_3 = self.encoder_block(input_channels=32, output_channels=64)
        self.encoder_4 = self.encoder_block(input_channels=64, output_channels=128)
        self.encoder_5 = self.encoder_block(input_channels=128, output_channels=256)

        self.decoder_1 = self.decoder_block(input_channels=256, output_channels=128)
        self.attention_1 = Attention_Block(input_channels=128, output_channels=128)
        self.up_conv_1 = self.encoder_block(input_channels=256, output_channels=128)

        self.decoder_2 = self.decoder_block(input_channels=128, output_channels=64)
        self.attention_2 = Attention_Block(input_channels=64, output_channels=64)
        self.up_conv_2 = self.encoder_block(input_channels=128, output_channels=64)

        self.decoder_3 = self.decoder_block(input_channels=64, output_channels=32)
        self.attention_3 = Attention_Block(input_channels=32, output_channels=32)
        self.up_conv_3 = self.encoder_block(input_channels=64, output_channels=32)

        self.decoder_4 = self.decoder_block(input_channels=32, output_channels=16)
        self.attention_4 = Attention_Block(input_channels=16, output_channels=16)
        self.up_conv_4 = self.encoder_block(input_channels=32, output_channels=16)

        self.final = self.final_block(16, 1)

    @staticmethod
    def final_block(input_channels, output_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=1),
            torch.nn.Sigmoid()
        )

    @staticmethod
    def decoder_block(input_channels, output_channels):
        return torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(inplace=True)
        )

    @staticmethod
    def encoder_block(input_channels, output_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        print(x.shape)
        x1 = self.encoder_1(x)

        x2 = self.max_pool(x1)
        x2 = self.encoder_2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encoder_3(x3)

        x4 = self.max_pool(x3)
        x4 = self.encoder_4(x4)

        x5 = self.max_pool(x4)
        x5 = self.encoder_5(x5)

        d5 = self.decoder_1(x5)
        x4 = self.attention_1(d5, x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_conv_1(d5)

        d4 = self.decoder_2(d5)
        x3 = self.attention_2(d4, x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv_2(d4)

        d3 = self.decoder_3(d4)
        x2 = self.attention_3(d3, x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv_3(d3)

        d2 = self.decoder_4(d3)
        x1 = self.attention_4(d2, x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv_4(d2)

        d1 = self.final(d2)
        return d1


def librosa_load(file_path, hop_length=256, n_fft=2048, sr_=16000):
    y, sr_ = librosa.load(file_path, sr=sr_)
    result = librosa.stft(y, hop_length=hop_length, n_fft=n_fft)
    return np.abs(result)


def process_audio(audio_tensor: torch.Tensor, max_time_: int, max_freq_: int) -> torch.Tensor:
    current_time = audio_tensor.shape[-1]
    if current_time < max_time_:
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, max_time_ - current_time))
    elif current_time > max_time_:
        audio_tensor = audio_tensor[:, :, :, :max_time_]

    current_freq = audio_tensor.shape[-2]
    if current_freq < max_freq_:
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, 0, 0, max_freq_ - current_freq))
    elif current_freq > max_freq_:
        audio_tensor = audio_tensor[:, :, :max_freq_, :]
    return audio_tensor


def compute_stats(dataset):
    all_data = []
    for i in range(len(dataset)):
        noisy_norm, _ = dataset[i]
        all_data.append(noisy_norm.numpy().flatten())

    all_data = np.concatenate(all_data)
    mean = np.mean(all_data)
    std = np.std(all_data)
    return mean, std


class AudioToDataset(torch.utils.data.Dataset):
    def __init__(self, clean_directory, noisy_directory):
        self.clean_files = sorted(Path(clean_directory).glob('*.wav'))
        self.noisy_files = sorted(Path(noisy_directory).glob('*.wav'))

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, item):
        clean_spectrogram = librosa_load(self.clean_files[item])
        noisy_spectrogram = librosa_load(self.noisy_files[item])
        return (
            torch.tensor(noisy_spectrogram).unsqueeze(0).float(),
            torch.tensor(clean_spectrogram).unsqueeze(0).float()
        )


def train(model: DenoisingUnet, epochs=50):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_func = torch.nn.SmoothL1Loss(beta=0.5)
    train_loss, val_loss = [], []
    # mae_loss, mse_loss = [], []
    patience = 7
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        for i, (noisy, clean) in enumerate(train_dataset):
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            predicted = model(noisy)
            loss = loss_func(predicted, clean)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                print(f'Batch: {i}, Train loss_func: {loss.item()}')
        avg_train_loss = torch.mean(torch.tensor(train_loss))
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for (noisy, clean) in validate_dataset:
                predicted = model(noisy)
                loss = loss_func(predicted, clean)
                val_loss.append(loss.item())
                epoch_val_loss += loss.item()
            avg_val_loss = epoch_val_loss / len(validate_dataset)
        avg_valid_loss = torch.mean(torch.tensor(val_loss))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Ранняя остановка активирована")
                break
        print(f'Epoch: {epoch}, Train Loss : {avg_train_loss}, Validate Loss : {avg_valid_loss}')


raw_train_dataset = AudioToDataset('datasets/train/clean', 'datasets/train/noisy')

max_time = max(spec[0].shape[-1] for spec in raw_train_dataset)
max_freq = max(spec[0].shape[-2] for spec in raw_train_dataset)


def to_even(x):
    return x if x % 2 == 0 else x + 1


def collate_fn(batch):
    processed_noisy = [process_audio(noisy, to_even(max_time), to_even(max_freq)) for noisy, clean in batch]
    processed_clean = [process_audio(clean, to_even(max_time), to_even(max_freq)) for noisy, clean in batch]
    return (
        torch.stack(processed_noisy),
        torch.stack(processed_clean)
    )


train_dataset = DataLoader(
    raw_train_dataset,
    shuffle=True,
    batch_size=4,
    collate_fn=collate_fn
)
validate_dataset = DataLoader(
    AudioToDataset('datasets/valid/clean', 'datasets/valid/noisy'),
    shuffle=True,
    batch_size=4,
    collate_fn=collate_fn
)


if __name__ == '__main__':
    model_ = DenoisingUnet()
    train(model_, 150)
    torch.save(model_.state_dict(), 'speech_denoising.pth')
