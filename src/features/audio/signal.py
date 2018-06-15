import librosa
from tqdm import tqdm
import json
import numpy as np

from audio.audioset import vggish_input, vggish_keras

np.random.seed(42)


def build_vggish_embeddings(dataset):
    paths = dataset['train']['path'] + dataset['test']['path']
    base_model = vggish_keras.get_vggish_keras()
    base_model.load_weights('src/features/audio/audioset/vggish_weights.ckpt')
    embeddings = []

    for path in tqdm(paths):
        y, sr = librosa.load(path, sr=8000)
        if len(y) < 8000:
            y = np.pad(y, (8000 - len(y)), 'constant')
        example = vggish_input.waveform_to_examples(y, 8000)
        embedding = base_model.predict(example[:, :, :, None])
        embeddings.append(embedding.tolist())

    with open('data/processed/signal.vggish.json', 'w') as f:
        json.dump(embeddings, f)
