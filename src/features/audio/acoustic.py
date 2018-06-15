import librosa
from tqdm import tqdm
import json
import scipy as sp
import numpy as np
np.random.seed(42)


def extract_features(path):

    stats = [np.mean, np.std, np.median, np.min,
             np.max, sp.stats.skew, sp.stats.kurtosis]

    row = []

    y, sr = librosa.audio.load(path, 8000)

    mfcc = np.nan_to_num(librosa.feature.mfcc(y, sr))
    zcr = np.nan_to_num(librosa.feature.zero_crossing_rate(y))
    rmse = np.nan_to_num(librosa.feature.rmse(y))
    chroma = np.nan_to_num(librosa.feature.chroma_stft(y, sr))
    spectral_centroid = np.nan_to_num(librosa.feature.spectral_centroid(y, sr))
    spectral_bandwidth = np.nan_to_num(
        librosa.feature.spectral_bandwidth(y, sr))
    spectral_flatness = np.nan_to_num(librosa.feature.spectral_flatness(y))
    spectral_rolloff = np.nan_to_num(librosa.feature.spectral_rolloff(y, sr))
    tonnetz = np.nan_to_num(librosa.feature.tonnetz(y, sr, chroma=chroma))

    for stat in stats:
        row += stat(mfcc, axis=1).tolist()
        row += stat(chroma, axis=1).tolist()
        row += stat(spectral_centroid, axis=1).tolist()
        row += stat(spectral_bandwidth, axis=1).tolist()
        row += stat(spectral_flatness, axis=1).tolist()
        row += stat(spectral_rolloff, axis=1).tolist()
        row += stat(tonnetz, axis=1).tolist()
    for stat in stats:
        row.append(float(stat(zcr, axis=1)))
        row.append(float(stat(rmse, axis=1)))

    row.append(len(y) / sr)

    return row


def build_features(dataset):
    paths = dataset['train']['path'] + dataset['test']['path']
    features = []

    for path in tqdm(paths):
        row = extract_features(path)
        features.append(row)

    with open('data/processed/acoustic.json', 'w') as f:
        json.dump(features, f)
