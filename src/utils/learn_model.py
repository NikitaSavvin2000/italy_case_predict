import os
import yaml
import shutil
import requests
import psycopg2
import numpy as np
import pandas as pd

from config import logger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Bidirectional, Dropout, Input, MaxPooling1D, Conv1D, Embedding,
                                     BatchNormalization, Reshape, Embedding, TimeDistributed, Flatten, Conv2D, GlobalAveragePooling2D)
from tensorflow.keras import regularizers


home_path = os.getcwd()

params_file = f'{home_path}/src/utils/params.yaml'


table_name = 'load_consumption'
measurement = 'load_consumption'

url_backend = os.getenv("BACKEND_URL", 'http://77.37.136.11:7070')

params_path = os.path.join(home_path, params_file)
params = yaml.load(open(params_path, 'r'), Loader=yaml.SafeLoader)
lstm0_units = params['lstm0_units']
lstm1_units = params['lstm1_units']
lstm2_units = params['lstm2_units']
regularizers_l2 = params["regularizers_l2"]
recurrent_dropout_rate = params["recurrent_dropout_rate"]
cnn0_units = params["cnn0_units"]
cnn1_units = params["cnn1_units"]

lag = params['lag']
activation = params['activation']
optimizer = params['optimizer']
epochs = params['epochs']
points_per_call = params['points_per_call']

points_to_predict = params['points_to_predict']

time_interval = 5
cols_for_train = []


DB_PARAMS = {
    "dbname": "mydb",
    "user": "myuser",
    "password": "mypassword",
    "host": "77.37.136.11",
    "port": 8083
}


def split_sequence(sequence, n_steps, horizon):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        out_end_ix = end_ix + horizon
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix:out_end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def normalization_request(col_time, col_target, json_list_df):

    url = f'{url_backend}/backend/v1/normalization'

    json = {"col_time": col_time, "col_target": col_target, "json_list_df": json_list_df}
    try:
        req = requests.post(
            url=url,
            json=json,
        )
        if req.status_code == 200:
            response_json = req.json()
            norm_df = pd.DataFrame.from_dict(response_json['df_all_data_norm'])
            min_val = float(response_json['min_val'])
            max_val = float(response_json['max_val'])

            return norm_df, min_val, max_val
        else:
            logger.error(f'Status code backend server: {req.status_code}')
            logger.error(req.status_code)
            return None, None, None
    except Exception as e:
        logger.error(e)
        return None, None, None


def reverse_normalization_request(col_time, col_target, json_list_norm_df, min_val, max_val):
    url = f'{url_backend}/backend/v1/reverse_normalization'
    json = {
        "col_time": col_time,
        "col_target": col_target,
        "min_val": min_val,
        "max_val": max_val,
        "json_list_norm_df": json_list_norm_df}
    try:
        req = requests.post(
            url=url,
            json=json,
        )
        if req.status_code == 200:
            reverse_de_norm_data_json = req.json()
            reverse_norm_df = pd.DataFrame.from_dict(reverse_de_norm_data_json['df_all_data_reverse_norm'])
            return reverse_norm_df
        else:
            logger.error(f'Status code backend server: {req.status_code}')
            logger.error(req.status_code)
            return None
    except Exception as e:
        logger.error(e)


def create_x_input(df_to_predict, n_steps):
    df_input = df_to_predict.iloc[len(df_to_predict) - n_steps:]
    x_input = df_input.values
    return x_input


def update_model():
    try:
        logger.info("Connecting to the database to fetch training data.")
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        select_query = f"""
        SELECT * FROM {table_name} ORDER BY datetime DESC;
        """
        logger.info(f"Executing SQL query: {select_query}")
        cur.execute(select_query)
        rows = cur.fetchall()

        if not rows:
            logger.error("No data retrieved from the database for training.")
            raise ValueError("No data retrieved from the database for training.")

        conn.commit()
        cur.close()
        conn.close()

        logger.info("Transforming data into DataFrame for training.")
        df_train = pd.DataFrame(rows, columns=["datetime", measurement])
        df_train["datetime"] = df_train["datetime"].dt.tz_localize(None)
        df_train["datetime"] = df_train["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

        print(df_train.head())
        df_train = df_train.sort_values(by='datetime', ascending=True)
        print(df_train.head())


        json_list_df_train = df_train.to_dict(orient='records')

        logger.info("Normalizing the training data.")
        df_train_norm, min_val, max_val = normalization_request(
            col_time='datetime',
            col_target=measurement,
            json_list_df=json_list_df_train
        )

        values = df_train_norm.values
        X, y = split_sequence(values, lag, points_per_call)

        logger.info("Building the LSTM model.")
        lstm_model = Sequential()

        lstm_model.add(LSTM(lstm0_units, activation='softplus', return_sequences=True, recurrent_dropout=recurrent_dropout_rate))
        lstm_model.add(LSTM(lstm1_units, activation=activation, return_sequences=True, recurrent_dropout=recurrent_dropout_rate))
        lstm_model.add(LSTM(lstm2_units, activation=activation, recurrent_dropout=recurrent_dropout_rate))

        lstm_model.add(Dense(points_per_call, activation='linear', kernel_regularizer=regularizers.l2(regularizers_l2)))

        lstm_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

        logger.info("Training the model.")
        lstm_model.fit(X, y, epochs=epochs, verbose=1)

        models_path = f'{home_path}/models'
        model_name = 'italy_case_model_2025.keras'
        saved_model_path = os.path.join(models_path, model_name)

        logger.info(f"Saving the model to {saved_model_path}.")
        lstm_model.save(saved_model_path)

        logger.info(f"Copying params file to {models_path}/params.yaml.")
        shutil.copy(params_file, os.path.join(models_path, "params.yaml"))

        logger.info("Model update and saving completed successfully.")

    except Exception as e:
        logger.error(f"Error during model update process: {e}")
        if conn:
            conn.rollback()
        raise e


update_model()