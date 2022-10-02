import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV


class Preprocessing:
    """
    Класс для препроцессинга, заполняет пропуски и бинаризует колонки с большим колличеством пропусков
    нормализует данные для линейных методов
    """

    def __init__(self, cat: list, num: list, is_linear: bool, fill_func: np.function_base = np.mean):
        """

        :param cat: категориальные признаки
        :param num: вещественные признаки
        :param is_linear: флаг, True для линейных методов
        :param fill_func: агрегирующая функция np, для заполнения пропусков в лин методах
        """

        self.cat = cat
        self.num = num
        self.is_linear = is_linear
        self.to_binarize = None

        self.fill_func = fill_func
        self.fill_values = None
        if is_linear:
            self.column_transformer = ColumnTransformer([
                ('ohe', OneHotEncoder(handle_unknown="ignore"), self.cat),
                ('scaling', StandardScaler(), self.num)
            ])
        else:
            self.column_transformer = ColumnTransformer([
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value',
                                           unknown_value=-1), self.cat),
                ('chill', 'passthrough', self.num)

            ])

    def fit_transform(self, df: pd.DataFrame):
        """

        :param df: датафрейм для обработки
        :return: догадайся
        """

        # бинаризуем скудные признаки
        self.to_binarize = np.array(df.columns)[(df.isna().sum() / df.shape[0] > 0.5).values]
        df[self.to_binarize] = df[self.to_binarize].isna().astype(int)
        # заполняем пропуски для вещественных признаков агрегатами, или -1000 для бустинга
        if self.is_linear:
            self.fill_values = dict(self.fill_func(df[self.num]))
            df[self.num] = df[self.num].fillna(value=self.fill_values)
        else:
            df[self.num] = df[self.num].fillna(value=-1000)
        # заполняем оставщееся пропуски
        df[self.cat] = df[self.cat].fillna(-1000)
        # OHE или Ordinal encoder
        df = self.column_transformer.fit_transform(df)
        return df

    def transform(self, df: pd.DataFrame):
        """
        Догадайся
        :param df: ты
        :return: можешь...
        """
        df[self.to_binarize] = df[self.to_binarize].isna().astype(int)
        df[self.cat] = df[self.cat].fillna(-1)
        df[self.num] = df[self.num].fillna(value=self.fill_values)
        df = self.column_transformer.transform(df)
        return df


class Pipe:
    """
    Класс для прогона моделей
    """

    def __init__(self, model, is_linear: bool):
        """

        :param model: модель поддерживающая fit и predict наследованный от склерновской базы
        :param is_linear: флаг - линейный алгоритм или нет
        """
        self.model = model
        self.final_model = None
        self.prep = None
        self.is_linear = is_linear

    def fit(self, x_train, y_train, params: dict, cat: list, num: list, cv: int = 5):
        """

        :param x_train:
        :param y_train:
        :param params: параметры для перебора гридсерчем
        :param cat: категориальные признаки
        :param num: вещественные признаки
        :param cv: колличество фолдов для кросс валидации
        :return: лучшая модель
        """
        self.prep = Preprocessing(cat=cat, num=num, is_linear=self.is_linear)
        x_train = self.prep.fit_transform(x_train)
        self.final_model = GridSearchCV(self.model,
                                        params,
                                        cv=cv,
                                        scoring='roc_auc',
                                        return_train_score=False,
                                        verbose=1,
                                        n_jobs=-1)
        self.final_model.fit(x_train, y_train)
        print('-----------------------------------------------')
        print(f'best_params is {self.final_model.best_params_}')
        print(f'best_score is {self.final_model.best_score_}')
        print()
        return self.final_model.best_estimator_

    def inference(self):
        return
