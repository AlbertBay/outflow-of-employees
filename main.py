import os
import pandas as pd

from time import time

from constants import DATA_FOLDER_NAME, DATAFRAMES, MODELS, PREDICT_FOLDER_NAME, VISUALISATION_FOLDER_NAME
from renew import Pipe, splitter


def model_run(df_name: str, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
              model: dict, storage_path: str, visualisation_path: str):
    """

    @param df_name:
    @param x_train:
    @param x_test:
    @param y_train:
    @param y_test:
    @param model:
    @param storage_path:
    @param visualisation_path:
    """
    pipe = Pipe(df_name,
                model['estimator_name'],
                model['model'],
                model['is_linear'],
                storage_path,
                visualisation_path)
    cat = x_train.select_dtypes(['object']).columns.tolist()
    num = x_train.select_dtypes(['int64', 'float64']).columns.tolist()
    pipe.fit(x_train, y_train, model['params'], cat, num)
    pipe.predict(x_test, y_test)


def run(data_folder: str = DATA_FOLDER_NAME, predict_folder: str = PREDICT_FOLDER_NAME,
        visualisation_folder: str = VISUALISATION_FOLDER_NAME):
    """

    :param data_folder:
    :param predict_folder:
    :param visualisation_folder:
    """
    path = os.getcwd()
    data_path = path + '/' + data_folder
    predict_path = path + '/' + predict_folder
    visualisation_path = path + '/' + visualisation_folder

    for filename in os.listdir(data_path):
        if filename.endswith('.csv'):
            print(f'Started working with {filename[:-4]} dataset.')
            df = pd.read_csv(data_path + '/' + filename)
            x_train, x_test, y_train, y_test = splitter(df,
                                                        DATAFRAMES[filename]['features'],
                                                        DATAFRAMES[filename]['y'])
            for model in MODELS:
                print(f'{MODELS[model]["estimator_name"]} has started to fit')
                model_run(filename[:-4], x_train, x_test, y_train, y_test, MODELS[model], predict_path, visualisation_path)

            print(f'Ended working with {filename[:-4]} dataset. Switching to next.')


if __name__ == '__main__':
    st_time = time()
    run()
    print(f'Total time spent: {(time() - st_time):.3f}s')
    print('Finish!')
