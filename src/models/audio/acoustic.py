import json
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import numpy as np

np.random.seed(42)
warnings.filterwarnings('ignore')


def get_features():
    with open('data/processed/acoustic.json', 'r') as f:
        features = json.load(f)
    return features


def logreg_gs():
    features = get_features()
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


def lightgbm_gs():
    features = get_features()
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


def xgboost_gs():
    features = get_features()
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
