from sklearn.dummy import DummyClassifier
import numpy as np


def get_features(dataset):
    return np.zeros((dataset['size'], 1))


def dummy_random_stratified(dataset):
    features = get_features(dataset)
    return DummyClassifier(strategy='stratified'), features


def dummy_most_frequent(dataset):
    features = get_features(dataset)
    return DummyClassifier(strategy='most_frequent'), features
