import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from typing import Callable, Optional

from constants import TEST_SIZE


def splitter(df: pd.DataFrame, features: list, y: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """

    @return:
    """
    x_train, x_test, y_train, y_test = train_test_split(df[features], df[y], test_size=TEST_SIZE)
    return x_train, x_test, y_train, y_test


class Preprocessing:
    """

    """
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
            self.column_transformer = ColumnTransformer([
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), self.cat),
                ('chill', 'passthrough', self.num)
            ])

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """

        @param df:
        @return:
        """
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
        df[self.to_binarize] = df[self.to_binarize].isna().astype(int)
        df[self.cat] = df[self.cat].fillna(-1)
        if self.is_linear:
            df[self.num] = df[self.num].fillna(value=self.fill_values)
        else:
            df[self.num] = df[self.num].fillna(value=-1000)
        df = self.column_transformer.transform(df)
        return df


class Pipe:
    """

    """
    dataframe_name: str
    estimator_name: str
    final_model: Optional[BaseEstimator]
    is_linear: bool
    model: BaseEstimator
    prep: Optional[Preprocessing]
    storage_path: str
    visualisation_path: str

    def __init__(self, dataframe_name: str, estimator_name: str, model: BaseEstimator, is_linear: bool,
                 storage_path: str, visualisation_path: str):
        """

        @param estimator_name:
        @param model:
        @param is_linear:
        @param storage_path:
        """
        self.dataframe_name = dataframe_name
        self.estimator_name = estimator_name
        self.model = model
        self.is_linear = is_linear
        self.storage_path = storage_path
        self.visualisation_path = visualisation_path

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, params: dict, cat: list, num: list, cv: int = 5):
        """

        @param x_train:
        @param y_train:
        @param params:
        @param cat:
        @param num:
        @param cv:
        """
        self.prep = Preprocessing(cat=cat, num=num, is_linear=self.is_linear)
        x_train = self.prep.fit_transform(x_train)
        final_model = GridSearchCV(self.model,
                                   params,
                                   cv=cv,
                                   scoring='roc_auc',
                                   return_train_score=False,
                                   verbose=1,
                                   n_jobs=-1,
                                   error_score='raise')
        final_model.fit(x_train, y_train)
        print('-----------------------------------------------------------------------------------------------------')
        print(f'best_params for {self.estimator_name} is {final_model.best_params_}')
        print(f'best_score for {self.estimator_name} is {final_model.best_score_}')
        print('-----------------------------------------------------------------------------------------------------\n')
        self.final_model = final_model.best_estimator_
        joblib.dump(self.final_model, self.storage_path + f'/{self.dataframe_name}_{self.estimator_name}_model.pkl')

    def predict(self, x_test: pd.DataFrame, y_test: pd.Series):
        """

        @param x_test:
        @param y_test:
        """
        x_test = self.prep.transform(x_test)
        y_pred_proba = self.final_model.predict_proba(x_test)[:, 1]
        y_pred_class = self.final_model.predict(x_test)
        print('-----------------------------------------------')
        print(f'ROC_AUC on test - {roc_auc_score(y_test, y_pred_proba)}')
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
