import os
import sys
import pandas as pd

from time import time
from warnings import simplefilter

from constants import AUTO_ML, DATA_FOLDER_NAME, DATAFRAMES, LOG_FOLDER_NAME, MODELS, PREDICT_FOLDER_NAME, \
    VISUALISATION_FOLDER_NAME
from pipes import Pipe, splitter, LAMA

simplefilter(action='ignore', category=FutureWarning)

sys.stdout = open('res.txt', 'w')

def model_run(df_name: str, x_train: pd.DataFrame, x_test: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.Series,
              y_test: pd.Series, y_val: pd.Series, model: dict, storage_path: str, visualisation_path: str):
    """

    @param df_name:
    @param x_train:
    @param x_test:
    @param y_train:
    @param y_test:
    @param model:
    @param storage_path:
    @param visualisation_path:
    @param y_val:
    @param x_val:
    """
    pipe = Pipe(df_name,
                model['estimator_name'],
                model['model'],
                model['is_linear'],
                storage_path,
                visualisation_path)
    cat = x_train.select_dtypes(['object']).columns.tolist()
    num = x_train.select_dtypes(['int64', 'float64']).columns.tolist()
    pipe.fit(x_train, y_train, x_val, y_val, model['params'], cat, num)
    pipe.predict(x_test, y_test)


def lama_run(df_name,
             train,
             val,
             test,
             target,
             storage_path,
             visualisation_path):
    lama = LAMA(df_name,
                estimator_name='lama',
                storage_path=storage_path,
                visualisation_path=visualisation_path
                )

    cat = train.drop(target, axis=1).select_dtypes(['object']).columns.tolist()
    num = train.drop(target, axis=1).select_dtypes(['int64', 'float64']).columns.tolist()

    lama.fit(train, val, cat, num, target)
    lama.predict(test)


def run(data_folder: str = DATA_FOLDER_NAME, log_folder: str = LOG_FOLDER_NAME,
        predict_folder: str = PREDICT_FOLDER_NAME, visualisation_folder: str = VISUALISATION_FOLDER_NAME):
    """

    :param data_folder:
    :param log_folder:
    :param predict_folder:
    :param visualisation_folder:
    :return:
    """
    path = os.getcwd()
    data_path = path + '/' + data_folder
    log_path = path + '/' + log_folder
    predict_path = path + '/' + predict_folder
    visualisation_path = path + '/' + visualisation_folder

    for filename in os.listdir(data_path):
        if filename.endswith('.csv'):
            print(f'Started working with {filename[:-4]} dataset.')
            df = pd.read_csv(data_path + '/' + filename)
            x_train, x_test, x_val, y_train, y_test, y_val = splitter(df,
                                                                      DATAFRAMES[filename]['features'],
                                                                      DATAFRAMES[filename]['y'])
            pd.concat([x_train, y_train], axis=1).to_csv(log_path + '/train_' + filename)
            pd.concat([x_test, y_test], axis=1).to_csv(log_path + '/test_' + filename)
            pd.concat([x_val, y_val], axis=1).to_csv(log_path + '/val_' + filename)

            for model in MODELS:
                print(f'{MODELS[model]["estimator_name"]}')
                model_run(filename[:-4],
                          x_train, x_test, x_val,
                          y_train, y_test, y_val,
                          MODELS[model],
                          predict_path,
                          visualisation_path)

            if AUTO_ML:
                print('LAMA processing')
                lama_run(filename[:-4],
                         x_train.assign(target=y_train.values),
                         x_val.assign(target=y_val.values),
                         x_test.assign(target=y_test.values),
                         target='target',
                         storage_path=predict_path,
                         visualisation_path=visualisation_path)
            print(f'Ended working with {filename[:-4]} dataset. Switching to next.')


if __name__ == '__main__':
    st_time = time()
    run()
    print(f'Total time spent: {(time() - st_time):.3f}s')
    print('Finish!')
