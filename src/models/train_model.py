# -*- coding: utf-8 -*-
import click
import logging
import sys
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
import numpy as np

from data.dataset import get_dataset

from models import baseline
from models.audio import signal, acoustic
from models.text import tfidf, bow

np.random.seed(42)


def build_train_test(dataset, features):
    X_train = np.array(features[:dataset['train_size']])
    X_test = np.array(features[dataset['train_size']:])
    y_train = np.array(dataset['train']['label'])
    y_test = np.array(dataset['test']['label'])
    return X_train, y_train, X_test, y_test


def report(model, dataset, features):
    logger = logging.getLogger(__name__)

    X_train, y_train, X_test, y_test = build_train_test(dataset, features)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    mae = mean_absolute_error(y_test, y_pred)

    logger.info({'acc': acc, 'f1_macro': f1_macro, 'mae': mae})


def report_gs(model, dataset, features):
    logger = logging.getLogger(__name__)

    X_train, y_train, X_test, y_test = build_train_test(dataset, features)
    gkf = list(GroupKFold(n_splits=3).split(
        X_train, y_train, dataset['train_groups']))
    model.cv = gkf
    grid_result = model.fit(X_train, y_train)

    logger.info("Best: %f using %s" %
                (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        logger.info("%f (+/-%f) with: %r" % (mean, stdev, param))

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    mae = mean_absolute_error(y_test, y_pred)

    logger.info({'acc': acc, 'f1_macro': f1_macro, 'mae': mae})


def evaluate(model, dataset):
    funcs = {
        'baseline.dummy_random_stratified': baseline.dummy_random_stratified,
        'baseline.dummy_most_frequent': baseline.dummy_most_frequent,
        'audio.signal.gaussiannb': signal.gaussiannb,
        'audio.signal.vggish_logreg_gs': signal.vggish_logreg_gs,
        'audio.signal.vggish_lightgbm_gs': signal.vggish_lightgbm_gs,
        'audio.signal.vggish_xgboost_gs': signal.vggish_xgboost_gs,
        'audio.signal.vggish_rnn_gs': signal.vggish_rnn_gs,
        'audio.acoustic.logreg_gs': acoustic.logreg_gs,
        'audio.acoustic.lightgbm_gs': acoustic.lightgbm_gs,
        'audio.acoustic.xgboost_gs': acoustic.xgboost_gs,
        'text.tfidf.logreg_gs': tfidf.logreg_gs,
        'text.tfidf.lightgbm_gs': tfidf.lightgbm_gs,
        'text.tfidf.xgboost_gs': tfidf.xgboost_gs,
        'text.bow.logreg_gs': bow.logreg_gs,
        'text.bow.lightgbm_gs': bow.lightgbm_gs,
        'text.bow.xgboost_gs': bow.xgboost_gs,
    }

    if model.endswith('_gs'):
        clf, features = funcs[model]()
        report_gs(clf, dataset, features)
    else:
        clf, features = funcs[model](dataset)
        report(clf, dataset, features)


@click.command()
@click.argument('model', type=str)
def main(model):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt=log_fmt)

    fh = logging.FileHandler('logs/model.train %s.log' % model, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info('Getting dataset')

    dataset = get_dataset()

    logger.info('Model evaluation')

    evaluate(model, dataset)

    logger.info('Done')


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
