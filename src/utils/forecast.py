import os
import yaml
import requests
import psycopg2
import numpy as np
import pandas as pd
import tensorflow as tf

from config import logger
from psycopg2.extras import execute_values
from tensorflow.keras.models import load_model


home_path = os.getcwd()


model_path = f'{home_path}/models/italy_case_model_2025.keras'

params_path = f'{home_path}/models/params.yaml'


table_name = 'load_consumption'
measurement = 'load_consumption'
table_predict_name = 'predict_load_consumption'

url_backend = os.getenv("BACKEND_URL", 'http://77.37.136.11:7070')

time_interval = 5

params = yaml.load(open(params_path, 'r'), Loader=yaml.SafeLoader)
lag = params['lag']
points_per_call = params['points_per_call']


DB_PARAMS = {
    "dbname": "mydb",
    "user": "myuser",
    "password": "mypassword",
    "host": "77.37.136.11",
    "port": 8083
}


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


def make_predictions(x_input, x_future, points_per_call, model):
    predict_values = []
    x_future_len = len(x_future)
    remaining_horizon = x_future_len

    if len(x_input.shape) == 2:
        x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))
    while remaining_horizon > 0:
        current_points_to_predict = min(remaining_horizon, points_per_call)
        x_input_tensor = tf.convert_to_tensor(x_input, dtype=tf.float32)
        y_predict = model.predict(x_input_tensor, verbose=0)

        if len(y_predict.shape) == 2 and y_predict.shape[0] == 1:
            y_predict = y_predict[0]

        y_predict = y_predict[:current_points_to_predict]
        predict_values.extend(y_predict)

        for i in range(current_points_to_predict):
            cur_val = y_predict[i]
            x_input = np.delete(x_input, (0), axis=1)
            future_lag = x_future[0]
            x_future = np.delete(x_future, 0, axis=0)
            future_lag[0] = cur_val
            x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)

        remaining_horizon -= current_points_to_predict

    return predict_values


def create_predict(count_time_points_predict):
    try:
        logger.info("Loading model.")
        model = load_model(model_path)

        logger.info(f"Connecting to database with limit={lag}")
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        select_query = f"""
        SELECT * FROM {table_name} ORDER BY datetime DESC LIMIT {count_time_points_predict};
        """

        logger.info(f"Executing SQL query: {select_query}")
        cur.execute(select_query)
        rows = cur.fetchall()

        if not rows:
            logger.error("No data retrieved from the database.")
            raise ValueError("No data retrieved from the database.")

        logger.info("Transforming data into DataFrame.")
        df_last_values = pd.DataFrame(rows, columns=["datetime", measurement])

        df_last_values["datetime"] = df_last_values["datetime"].dt.tz_localize(None)

        df_last_values = df_last_values.sort_values(by='datetime', ascending=True)

        last_know_date = df_last_values["datetime"].iloc[-1]
        logger.info(f"Last known date: {last_know_date}")

        datetime_range = pd.date_range(
            start=last_know_date,
            periods=count_time_points_predict,
            freq=f"{time_interval}T"
        ).floor("T")

        df_predict = pd.DataFrame({"datetime": datetime_range, "load_consumption": None})

        frames = [df_last_values, df_predict]
        general_df = pd.concat(frames)

        general_df = general_df.fillna("None")
        general_df["datetime"] = general_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

        json_list_general_norm_df = general_df.to_dict(orient='records')

        logger.info("Normalizing the data.")
        df_general_norm_df, min_val, max_val = normalization_request(
            col_time='datetime',
            col_target=measurement,
            json_list_df=json_list_general_norm_df
        )

        df_general_norm_df = df_general_norm_df.replace("None", None)
        df_to_predict_norm = df_general_norm_df.iloc[:-count_time_points_predict]
        df_predict_norm = df_general_norm_df.iloc[-count_time_points_predict:]

        print(df_to_predict_norm.columns)

        df_to_predict_norm = df_to_predict_norm.drop(columns=['datetime'])
        df_predict_norm = df_predict_norm.drop(columns=['datetime'])

        values = df_to_predict_norm.values
        n_features = values.shape[1]
        x_input = create_x_input(df_to_predict_norm, lag)
        x_input = x_input.reshape((1, lag, n_features))

        x_future = df_predict_norm.values
        logger.info(f"Making predictions for {len(x_future)} future points.")

        predict_values = make_predictions(x_input, x_future, points_per_call, model)

        df_predict_norm[measurement] = predict_values

        json_list_df_predict_norm = df_predict_norm.to_dict(orient='records')

        logger.info("Reversing normalization on predictions.")

        df_predict = reverse_normalization_request(
            col_time='datetime',
            col_target=measurement,
            json_list_norm_df=json_list_df_predict_norm,
            min_val=min_val,
            max_val=max_val
        )
        print('is working 6')

        data = [(row['datetime'], row[measurement]) for _, row in df_predict.iterrows()]

        query = f"""
            INSERT INTO {table_predict_name} (datetime, {measurement})
            VALUES %s
            ON CONFLICT (datetime)
            DO UPDATE SET {measurement} = EXCLUDED.{measurement};
        """

        logger.info(f"Inserting forecasted data into the database.")
        execute_values(cur, query, data)

        conn.commit()
        cur.close()
        conn.close()

        logger.info('Data successfully predicted and inserted into the database.')

    except Exception as e:
        logger.error(f"Error during prediction process: {e}")
        if conn:
            conn.rollback()
        raise e
