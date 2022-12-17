from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

DATA_FOLDER_NAME = 'data'

LOG_FOLDER_NAME = 'log'

PREDICT_FOLDER_NAME = 'predict'

VISUALISATION_FOLDER_NAME = 'visualisation'

TEST_SIZE = 0.3
# использовать ли LAMA
AUTO_ML = True
# метрика для отбора моделей должна быть либо 'f1' либо 'roc_auc'
METRIC_TO_SEARCH = 'f1'
# параметры р
CALIBRATION_STATUS = True
LAMB = 0.58
# фиксированный random seed
RANDOM_SEED = 42

MODELS = {

    0: {'estimator_name': 'CatBoost',
        'is_linear': False,
        'params': {'learning_rate': [0.1, 0.5, 1], 'iterations': [100, 200, 500], 'depth': [5, 10],
                   'thread_count': [-1]},
        'model': CatBoostClassifier(verbose=False, random_state=RANDOM_SEED)},
    1: {'estimator_name': 'LGBM',
        'is_linear': False,
        'params': {'learning_rate': [0.1, 0.5, 1], 'n_estimators': [100, 200, 500], 'max_depth': [5, 10],
                   'n_jobs': [-1]},
        'model': LGBMClassifier(random_state=RANDOM_SEED)},
    2: {'estimator_name': 'Decision tree',
        'is_linear': False,
        'params': {'max_depth': [5, 10, 15, 30]},
        'model': DecisionTreeClassifier(random_state=RANDOM_SEED)},
    3: {'estimator_name': 'Random Forest',
        'is_linear': False,
        'params': {'n_estimators': [100, 500, 1000], 'max_depth': [5, 10, 15, 30],
                   'n_jobs': [-1]},
        'model': RandomForestClassifier(random_state=RANDOM_SEED)},
    4: {'estimator_name': 'KNN',
        'is_linear': True,
        'params': {'n_neighbors': [5, 10, 15],
                   'n_jobs': [-1]},
        'model': KNeighborsClassifier()},
    5: {'estimator_name': 'LogReg',
        'is_linear': True,
        'params': {'C': [1.0, 0.5, 0.2], 'max_iter': [100, 200, 500],
                   'n_jobs': [-1]},
        'model': LogisticRegression(random_state=RANDOM_SEED)},
    6: {'estimator_name': 'SVC',
        'is_linear': True,
        'params': {'C': [1.0, 0.5, 0.2], 'kernel': ['linear', 'poly']},
        'model': SVC(probability=True, random_state=RANDOM_SEED)},
    7: {'estimator_name': 'Naive Bayes',
        'is_linear': True,
        'params': {'priors': [None]},
        'model': GaussianNB()},

}
DATAFRAMES = {
    'turnover.csv': {'features': ['stag',
                                  'gender',
                                  'age',
                                  'industry',
                                  'profession',
                                  'traffic',
                                  'coach',
                                  'head_gender',
                                  'greywage',
                                  'way',
                                  'extraversion',
                                  'independ',
                                  'selfcontrol',
                                  'anxiety',
                                  'novator'],
                     'y': 'event'},
}
