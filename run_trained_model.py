def run_trained_model(X_unseen):
  import os
  import numpy as np
  import torch
  import torch.nn as nn
  import torchaudio
  import gdown
  from torch.utils.data import DataLoader, Dataset
  from torchvision.transforms import Compose
  import torchaudio.transforms as T

  def download_model_weights():
    import gdown
    url = 'https://drive.google.com/file/d/1w-4zPuC1JhCt2lEWxm0KbAOewCchuKZ8/view?usp=sharing'
    output = "my_weights.pth"
    gdown.download(url, output, fuzzy=True)
    return output
  weight_path = download_model_weights()
  weights = np.load(weight_path, allow_pickle=True)

  class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

  class CNN14(nn.Module):
    def __init__(self, num_classes):
        super(CNN14, self).__init__()

        # 6个卷积块及其后面的平均池化层
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.conv5 = ConvBlock(512, 1024)
        #self.conv6 = ConvBlock(1024, 2048)

        # 自适应全局平均池化，将空间维度强行变为1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.adaptive_pool_mid = nn.AdaptiveAvgPool2d((8, 8))

        # 最终的全连接层，输入特征维度 2048 -> 输出类别数
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.adaptive_pool_mid(x)
        x = self.conv2(x)
        x = self.adaptive_pool_mid(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #x = self.conv6(x)

        # 自适应池化到1x1
        x = self.adaptive_pool(x)

        # 展平成(batch, 2048)
        x = x.view(x.size(0), -1)

        # 全连接层分类
        x = self.fc(x)
        return x


  class AudioDataset(Dataset):
        def __init__(self, filepaths, labels, transform=None):
            self.filepaths = filepaths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.filepaths)

        def __getitem__(self, idx):
            filepath = self.filepaths[idx]
            label = self.labels[idx]

            waveform, _ = torchaudio.load(filepath)
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if self.transform:
                waveform = self.transform(waveform)
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)
            return waveform, label

  def transform_audio(sample_rate=16000, n_fft=512, n_mels=64, hop_length=256):
    return Compose([
        # 重采样到目标频率
        T.Resample(orig_freq=sample_rate, new_freq=16000),

        # 计算 Mel Spectrogram
        T.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        ),

        # 转换为分贝单位
        T.AmplitudeToDB(),

        # 计算 Delta 和 Delta-Delta
        DeltaAndDeltaDelta()  # 自定义变换，添加 Delta 和 Delta-Delta 特征
    ])


  # 自定义变换，添加 Delta 和 Delta-Delta 特征
  class DeltaAndDeltaDelta:
    def __call__(self, mel_db):
        delta = T.ComputeDeltas()(mel_db)           # 计算一阶导数
        delta2 = T.ComputeDeltas()(delta)           # 计算二阶导数

        # 拼接 mel_db, delta 和 delta2 特征 (channel 维度)
        combined_features = torch.cat([mel_db, delta, delta2], dim=0)
        return combined_features

  def collate_fn(batch):
        waveforms, labels = zip(*batch)
        max_len = max(waveform.shape[-1] for waveform in waveforms)
        padded_waveforms = [torch.nn.functional.pad(waveform, (0, max_len - waveform.shape[-1])) for waveform in waveforms]
        return torch.stack(padded_waveforms), torch.tensor(labels, dtype=torch.long)

  def classifier(X_unseen, Y_unseen):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights_path = download_model_weights()
        #weights = np.load(weight_path, allow_pickle=True)

        #weights_path = "best_model.pth"

        # Load model
        model = CNN14(num_classes=7).to(device)
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # Prepare data and predict
        transform = transform_audio()
        val_dataset = AudioDataset(X_unseen, Y_unseen, transform=transform)
        eval_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
        all_preds = []

        with torch.no_grad():
            for batch_waveforms, _ in eval_loader:
                batch_waveforms = batch_waveforms.to(device)
                outputs = model(batch_waveforms)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())

        return np.array(all_preds)

  predictions = classifier(X, Y)
  assert predictions.shape == Y.shape
  return predictions
