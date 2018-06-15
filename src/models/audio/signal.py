import librosa
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
import json

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasClassifier

import scipy as sp
import numpy as np
np.random.seed(42)


def get_signals(dataset):
    paths = dataset['train']['path'] + dataset['test']['path']
    signals = []

    for path in tqdm(paths):
        y, sr = librosa.audio.load(path, 8000)
        signals.append(y)

    return signals


def pad_signals(signals, pad_len=5000):
    padded = pad_sequences(signals, maxlen=pad_len,
                           truncating='pre', dtype='float32')
    return padded


def get_vggish_embeddings():
    with open('data/processed/signal.vggish.json', 'r') as f:
        embeddings = json.load(f)
    return embeddings


def get_vggish_features():
    embeddings = get_vggish_embeddings()
    features = []
    for emb in tqdm(embeddings):
        vec = np.concatenate((np.mean(emb, axis=0),
                              np.min(emb, axis=0),
                              np.max(emb, axis=0),
                              np.std(emb, axis=0),
                              np.median(emb, axis=0),
                              sp.stats.skew(emb, axis=0),
                              sp.stats.kurtosis(emb, axis=0)))
        features.append(vec)

    return features


def gaussiannb(dataset):
    signals = get_signals(dataset)
    features = pad_signals(signals)
    gnb = GaussianNB()
    return gnb, features


def vggish_logreg_gs():
    features = get_vggish_features()
    lr_pl = Pipeline([('ss', StandardScaler()), ('lr', LogisticRegression())])
    params = {
        'lr__tol': [1e-4, 1e-3, 1e-2],
        'lr__C': [1e-2, 0.1, 1, 10],
        'lr__solver': ['newton-cg', 'sag', 'saga', 'lbfgs']
    }
    lr_gs = GridSearchCV(lr_pl, params, verbose=2,
                         error_score=0, scoring='neg_mean_absolute_error',
                         n_jobs=30)
    return lr_gs, features


def vggish_lightgbm_gs():
    features = get_vggish_features()
    lgbm = LGBMClassifier(n_jobs=30)
    params = {
        'n_estimators': [200],
        'learning_rate': [0.03, 0.1, 0.3],
        'subsample': [0.3, 0.5, 0.7, 1],
        'colsample_bytree': [0.3, 0.5, 0.7, 1]
    }
    lgbm_gs = GridSearchCV(lgbm, params, verbose=2,
                           error_score=0, scoring='neg_mean_absolute_error')
    return lgbm_gs, features


def vggish_xgboost_gs():
    features = get_vggish_features()
    xgb = XGBClassifier(nthread=30)
    params = {
        'n_estimators': [200],
        'max_depth': [3, 5, 10],
        'min_child_weight': [0.1, 0.3, 0.5, 0.7, 1],
        'subsample': [0.5, 0.7, 1],
        'scale_pos_weight': [0.5, 1, 3, 10]
    }
    xgb_gs = GridSearchCV(xgb, params, verbose=2,
                          error_score=0, scoring='neg_mean_absolute_error')
    return xgb_gs, features


def vggish_rnn_gs():
    embeddings = get_vggish_embeddings()
    maxlen = max([len(emb) for emb in embeddings])
    features = pad_sequences(embeddings, maxlen=maxlen, dtype='float32')

    def create_model(neurons, learning_rate):
        model = Sequential()
        model.add(LSTM(128, input_shape=(None, 128)))
        model.add(Dense(neurons))
        model.add(Dense(neurons))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=learning_rate),
                      metrics=['accuracy'])
        return model

    model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=256)
    params = {
        'neurons': [64, 128, 256],
        'learning_rate': [1e-3, 1e-4]
    }
    rnn_gs = GridSearchCV(model, params, verbose=2,
                          error_score=0, scoring='neg_mean_absolute_error')
    return rnn_gs, features
