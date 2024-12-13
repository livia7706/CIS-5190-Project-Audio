def run_trained_model(X):
  # X is of shape (N, ), where each element is a path string to a WAV file.
  # TODO: featurize the WAV files
  !pip install git+https://github.com/openai/whisper.git
  import whisper

  import tensorflow as tf
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
  import librosa
  import numpy as np

  # Step 1: Featurize the WAV files
  def load_and_resample(file_path, target_sr=16000):
      audio, sr = librosa.load(file_path, sr=target_sr)  # Resample to target_sr
      return audio

  def extract_features(audio):
      import whisper
      mel = whisper.log_mel_spectrogram(audio).detach().numpy()
      return mel

  # Process input X to extract features

  features = []
  for file_path in X:
    if file_path.endswith('.wav'):
      audio = load_and_resample(file_path)
      feature = extract_features(audio)
      features.append(feature)
  # truncate features to match target_length
  lengths = [feature.shape[1] for feature in features]
  target_length = 100
  #target_length = int(np.median(lengths))
  features_fixed = [
      feature[:, :target_length] if feature.shape[1] > target_length else
      np.pad(feature, ((0, 0), (0, target_length - feature.shape[1])), 'constant')
      for feature in features
  ]
  features_array = np.array(features_fixed).reshape(len(features_fixed), 80, target_length, 1)


  # TODO: load your model weights
  def download_model_weights():
    import gdown
    url = 'https://drive.google.com/file/d/15SurM8a68wJtIDH5MyYjOYr9KCA2-0EM/view?usp=sharing'
    output = "best_weights.weights.h5"
    gdown.download(url, output, fuzzy=True)

    print(f"Downloaded file size: {os.path.getsize(output)} bytes")

    return output

  weight_path = download_model_weights()
  #weight_path = '/content/drive/My Drive/project/my_weights.weights.h5'
  #weights = np.load(weight_path, allow_pickle=True)


  # TODO: setup model
  model = Sequential([
      Conv2D(32, (3, 3), activation='relu', input_shape=(80, target_length, 1)),
      MaxPooling2D((2, 2)),
      Conv2D(64, (3, 3), activation='relu'),
      MaxPooling2D((2, 2)),
      Flatten(),
      Dense(128, activation='relu'),
      Dropout(0.5),
      Dense(7, activation='softmax')
  ])
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  #model.summary()
  model.load_weights(weight_path)
  #model.summary()


  predictions = []

  pred = model.predict(features_array)
  #predictions.append(np.argmax(pred, axis=1))
  predictions = np.argmax(pred, axis=1)

  predictions = np.array(predictions) # Should be shape (N,) where each element is a class integer for the corresponding data point.
  print(predictions.shape)
  print(Y.shape)
  assert predictions.shape == Y.shape
  return predictions
