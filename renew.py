import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, f1_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from typing import Callable, Optional

from constants import TEST_SIZE
from constants import METRIC_TO_SEARCH
from constants import CALIBRATION_STATUS
from constants import lamb

warnings.filterwarnings("ignore")


def splitter(df: pd.DataFrame, features: list, y: str) -> tuple[pd.DataFrame,
                                                                pd.DataFrame,
                                                                pd.DataFrame,
                                                                pd.Series,
                                                                pd.Series,
                                                                pd.Series]:
    """

    @return:
    """
    x_train, x_test, y_train, y_test = train_test_split(df[features], df[y], test_size=TEST_SIZE, stratify=df[y])
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=TEST_SIZE, stratify=y_train)
    return x_train, x_test, x_val, y_train, y_test, y_val


def f1(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    return f1_score(y_true, (y_pred > 0.5) * 1)


METRICS_FUNC = {'f1': f1, 'roc_auc': roc_auc_score}


class Preprocessing:
    cat: list
    column_transformer: ColumnTransformer
    fill_func: Callable
    fill_values: dict
    is_linear: bool
    num: list
    to_binarize: Optional[np.array]

    def __init__(self, cat: list, num: list, is_linear: bool, fill_func: Callable = np.mean):
        """

        @param cat:
        @param num:
        @param is_linear:
        @param fill_func:
        """
        self.cat = cat
        self.num = num
        self.is_linear = is_linear
        self.fill_func = fill_func
        self.to_binarize = None

        if is_linear:
            self.column_transformer = ColumnTransformer([
                ('ohe', OneHotEncoder(handle_unknown="ignore"), self.cat),
                ('scaling', StandardScaler(), self.num)
            ])
        else:
            print(f'Categorical features are {self.cat}')
            print(f'Numeric features are {self.num}')
            self.column_transformer = ColumnTransformer([
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), self.cat),
                ('chill', 'passthrough', self.num)
            ])

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """

        @param df:
        @return:
        """
        df[self.cat] = df[self.cat].astype(str)
        df[self.num] = df[self.num].astype(float)
        self.to_binarize = np.array(df.columns)[(df.isna().sum() / df.shape[0] > 0.5).values]
        df[self.to_binarize] = df[self.to_binarize].isna().astype(int)
        if self.is_linear:
            self.fill_values = dict(self.fill_func(df[self.num]))
            df[self.num] = df[self.num].fillna(value=self.fill_values)
        else:
            df[self.num] = df[self.num].fillna(value=-1000)
        df[self.cat] = df[self.cat].fillna(-1000)
        df = self.column_transformer.fit_transform(df)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """

        @param df:
        @return:
        """
        df[self.cat] = df[self.cat].astype(str)
        df[self.num] = df[self.num].astype(float)
        df[self.to_binarize] = df[self.to_binarize].isna().astype(int)
        df[self.cat] = df[self.cat].fillna(-1)
        if self.is_linear:
            df[self.num] = df[self.num].fillna(value=self.fill_values)
        else:
            df[self.num] = df[self.num].fillna(value=-1000)
        df = self.column_transformer.transform(df)
        return df


class Pipe:
    def __init__(self, dataframe_name: str, estimator_name: str, model: BaseEstimator, is_linear: bool,
                 storage_path: str, visualisation_path: str, metric: str = METRIC_TO_SEARCH,
                 calibration_status: bool = CALIBRATION_STATUS):
        """

        @param estimator_name:
        @param model:
        @param is_linear:
        @param metric:
        @param storage_path:
        """

        self.dataframe_name = dataframe_name
        self.estimator_name = estimator_name
        self.model = model
        self.is_linear = is_linear
        self.storage_path = storage_path
        self.visualisation_path = visualisation_path
        self.threshold = 0.5
        self.metric = metric
        self.calibration_status = calibration_status
        self.cal_model = None
        self.prep = None
        self.final_model = None

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series,
            x_val: pd.DataFrame, y_val: pd.Series,
            params: dict, cat: list, num: list, cv: int = 5):
        """

        @param x_train:
        @param y_train:
        @param params:
        @param cat:
        @param num:
        @param cv:
        @param x_val:
        @param y_val:
        """

        self.prep = Preprocessing(cat=cat, num=num, is_linear=self.is_linear)
        x_train = self.prep.fit_transform(x_train)
        x_val = self.prep.transform(x_val)
        # очень плохо, но что делать
        if isinstance(self.model, GaussianNB) & (not isinstance(x_train, np.ndarray)):
            x_train = x_train.todense()
        if isinstance(self.model, GaussianNB) & (not isinstance(x_val, np.ndarray)):
            x_val = x_val.todense()
        final_model = GridSearchCV(self.model,
                                   params,
                                   cv=cv,
                                   scoring=METRIC_TO_SEARCH,
                                   return_train_score=False,
                                   verbose=1,
                                   n_jobs=-1,
                                   error_score='raise')
        final_model.fit(x_train, y_train)

        print('------------------------------------TRAINING INFO-------------------------------------------------')
        print(f'best_params for {self.estimator_name} is {final_model.best_params_}')
        print(f'best_{METRIC_TO_SEARCH}_score on train for {self.estimator_name} is {final_model.best_score_}')
        preds = final_model.predict_proba(x_val)[:, 1]

        if self.calibration_status:
            print('starting calibration')
            w1 = (lamb * np.sum(y_val == 0)) / ((1 - lamb) * np.sum(y_val == 1))
            cal_model = LogisticRegression(random_state=42,
                                           class_weight={0: 1, 1: w1}
                                           )
            cal_model.fit(X=preds.reshape(-1, 1), y=y_val)
            preds = cal_model.predict_proba(preds.reshape(-1, 1))[:, 1]
            self.cal_model = cal_model

        precision, recall, thresholds = precision_recall_curve(y_val, preds)
        fscore = (2 * precision * recall) / (precision + recall)
        fscore[np.isnan(fscore)] = -1  # for infinity
        ix = np.argmax(fscore)

        print('Best Threshold is %f with F-Score=%.3f on validation' % (thresholds[ix], fscore[ix]))

        self.threshold = thresholds[ix]
        self.final_model = final_model.best_estimator_
        joblib.dump(self, self.storage_path + f'/{self.dataframe_name}_{self.estimator_name}_model.pkl')

    def predict(self, x_test: pd.DataFrame, y_test: pd.Series):
        """

        @param x_test:
        @param y_test:
        """
        x_test = self.prep.transform(x_test)
        # очень плохо, но что делать
        if isinstance(self.model, GaussianNB) & (not isinstance(x_test, np.ndarray)):
            x_test = x_test.todense()
        y_pred_proba = self.final_model.predict_proba(x_test)[:, 1]

        if self.calibration_status:
            y_pred_proba = self.cal_model.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]

        y_pred_class = (y_pred_proba > self.threshold) * 1
        print('------------------------------------TESTING INFO-------------------------------------------------')
        print(f'Roc_Auc on test - {roc_auc_score(y_test, y_pred_proba)}')
        print(f'F1 on test - {f1_score(y_test, y_pred_class)}')
        print(classification_report(y_test, y_pred_class, target_names=['class 0', 'class 1']))

        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(10, 8), dpi=100)
        plt.axis('scaled')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title("AUC & ROC Curve")
        plt.plot(false_positive_rate, true_positive_rate, 'g')
        plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig(self.visualisation_path + f'/{self.dataframe_name}_{self.estimator_name}_ROC_AUC.jpg')

        figure, axis = plt.subplots(1, 2, figsize=(10, 15))
        axis[0].hist(y_pred_proba[y_test == 1])
        axis[0].set_title('1 CLASS')
        axis[1].hist(y_pred_proba[y_test == 0])
        axis[1].set_title('0 CLASS')
        plt.savefig(self.visualisation_path + f'/{self.dataframe_name}_{self.estimator_name}_hist.jpg')
        print('-----------------------------------------------------------------------------------------------------\n')


class LAMA:
    def __init__(self, dataframe_name: str, estimator_name: str,
                 storage_path: str, visualisation_path: str, metric: str = METRIC_TO_SEARCH,
                 calibration_status: bool = CALIBRATION_STATUS):
        """

        :param dataframe_name:
        :param estimator_name:
        :param storage_path:
        :param visualisation_path:
        :param metric:
        """

        self.metric = f1 if metric == 'f1' else 'auc'
        self.task = Task(name='binary', metric=self.metric, greater_is_better=True)
        self.final_model = TabularAutoML(task=self.task, timeout=2000, reader_params={'cv': 5, 'random_state': 42})
        self.roles = None
        self.dataframe_name = dataframe_name
        self.estimator_name = estimator_name
        self.storage_path = storage_path
        self.visualisation_path = visualisation_path
        self.threshold = 0.5
        self.calibration_status = calibration_status
        self.cal_model = None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, cat: list, num: list, target: str):
        """

        :param target:
        :param train:
        :param val:
        :param cat:
        :param num:
        :return:
        """

        self.roles = {
            'target': target,
            'category': cat,
            'numeric': num
        }

        oof_pred = self.final_model.fit_predict(
            train,
            roles=self.roles
        )

        print('------------------------------------TRAINING INFO-------------------------------------------------')
        metric_on_train = METRICS_FUNC[METRIC_TO_SEARCH](train[self.roles["target"]], oof_pred.data[:, 0])
        print(f'best_{METRIC_TO_SEARCH}_score on train for LAMA is {metric_on_train}')
        preds = self.final_model.predict(val).data[:, 0]
        if self.calibration_status:
            print('starting calibration')
            w1 = (lamb * np.sum(val[target] == 0)) / ((1 - lamb) * np.sum(val[target] == 1))
            cal_model = LogisticRegression(random_state=42, class_weight={0: 1, 1: w1})
            cal_model.fit(np.array(preds).reshape(-1, 1), val[target])

            preds = cal_model.predict_proba(np.array(preds).reshape(-1, 1))[:, 1]
            self.cal_model = cal_model

        precision, recall, thresholds = precision_recall_curve(val[target], preds)
        fscore = (2 * precision * recall) / (precision + recall)
        fscore[np.isnan(fscore)] = -1  # for infinity
        ix = np.argmax(fscore)
        print('Best Threshold is %f with F-Score=%.3f on validation' % (thresholds[ix], fscore[ix]))
        self.threshold = thresholds[ix]
        joblib.dump(self, self.storage_path + f'/{self.dataframe_name}_{self.estimator_name}_model.pkl')

    def predict(self, test: pd.DataFrame):
        """

        :param test:
        :return:
        """

        y_pred_proba = self.final_model.predict(test).data[:, 0]
        if self.calibration_status:
            y_pred_proba = self.cal_model.predict_proba(np.array(y_pred_proba).reshape(-1, 1))[:, 1]
        y_pred_class = (y_pred_proba > self.threshold) * 1
        print('------------------------------------TESTING INFO-------------------------------------------------')
        print(f'Roc_auc on test - {roc_auc_score(test[self.roles["target"]], y_pred_proba)}')
        print(f'F1 on test - {f1_score(test[self.roles["target"]], y_pred_class)}')
        print(classification_report(test[self.roles["target"]], y_pred_class, target_names=['class 0', 'class 1']))

        false_positive_rate, true_positive_rate, thresholds = roc_curve(test[self.roles["target"]], y_pred_proba)
        plt.figure(figsize=(10, 8), dpi=100)
        plt.axis('scaled')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title("AUC & ROC Curve")
        plt.plot(false_positive_rate, true_positive_rate, 'g')
        plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig(self.visualisation_path + f'/{self.dataframe_name}_{self.estimator_name}_ROC_AUC.jpg')

        figure, axis = plt.subplots(1, 2, figsize=(10, 15))
        axis[0].hist(y_pred_proba[test[self.roles["target"]] == 1])
        axis[0].set_title('1 CLASS')
        axis[1].hist(y_pred_proba[test[self.roles["target"]] == 0])
        axis[1].set_title('0 CLASS')
        plt.savefig(self.visualisation_path + f'/{self.dataframe_name}_{self.estimator_name}_hist.jpg')
        print('-----------------------------------------------------------------------------------------------------\n')
