def run_trained_model(X):
  # X is of shape (N, ), where each element is a path string to a WAV file.
  # TODO: featurize the WAV files
  !pip install torchvision torchaudio timm
  import torchaudio.transforms as T
  import torch
  import random
  from sklearn.model_selection import train_test_split
  import torch.nn as nn
  from timm import create_model
  import os
  import torchaudio
  from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
  from torchvision.transforms import Resize
  import torch
  import numpy as np
  from torch.utils.data import Dataset

  def preprocess_audio(file_path):

    # mel transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()
    resize_transform = Resize((300, 300))  # resize to 300 * 300

    # load file
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    mel_spec = mel_transform(waveform)
    mel_spec_db = db_transform(mel_spec)
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    mel_spec_db = resize_transform(mel_spec_db)

    if mel_spec_db.ndim == 2:  # if (H, W) -> (1, H, W)
        mel_spec_db = mel_spec_db.unsqueeze(0)
    elif mel_spec_db.ndim == 3 and mel_spec_db.shape[0] == 1:  # if (1, H, W)
        pass
    else:
        raise ValueError(f"Unexpected shape of mel_spec_db: {mel_spec_db.shape}")

    mel_spec_db = mel_spec_db.repeat(3, 1, 1)  #  (1, H, W) -> (3, H, W)
    return mel_spec_db

  features = []

  for file_path in X:
      mel_spec_db = preprocess_audio(file_path)
      features.append(mel_spec_db.numpy())

  # To NumPy array
  features = np.array(features)
  print(f"Loaded {len(features)} samples.")


  # (N, 3, H, W)
  if features.ndim == 4 and features.shape[1] == 1:  # (N, 1, H, W)
      features = features.repeat(1, 3, 1, 1)  


  # TODO: load your model weights
  def download_model_weights():
    import gdown
    url = 'https://drive.google.com/file/d/1VQxBJhKYflzFAWkHmuXcVVJu9fwETkfM/view?usp=sharing'
    output = "my_weights.pth"
    gdown.download(url, output, fuzzy=True)
    return output
  weight_path = download_model_weights()
  #weights = np.load(weight_path, allow_pickle=True)
  weights = torch.load(weight_path, map_location='cpu')

  # TODO: setup model
  model_name = "efficientnet_b3"  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = create_model(model_name, pretrained=False, num_classes=len(CLASS_TO_LABEL))
  model.load_state_dict(weights)   
  model = model.to(device)

  features = torch.tensor(features).float().to(device)

  model.eval()
  predictions = []
  with torch.no_grad():
      outputs = model(features)
      _, predicted = torch.max(outputs, 1)
      predictions = predicted.cpu().numpy() # Should be shape (N,) where each element is a class integer for the corresponding data point.

  assert predictions.shape == Y.shape
  return predictions
