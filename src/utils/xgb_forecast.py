import os
import json
import requests
import psycopg2
import pandas as pd

from config import logger
from psycopg2.extras import execute_values



table_name = 'load_consumption'
measurement = 'load_consumption'
table_predict_name = 'xgb_predict_load_consumption'


model_architecture_params = {
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "learning_rate": 0.1,
    "max_depth": 15,
    "subsample": .9,
    "colsample_bytree": .9,
    "min_child_weight": 5,
    "booster": "gbtree"

}

DB_PARAMS = {
    "dbname": "mydb",
    "user": "myuser",
    "password": "mypassword",
    "host": "77.37.136.11",
    "port": 8083
}

time_interval = 5
lag = 4

url_backend = os.getenv("BACKEND_URL", 'http://77.37.136.11:7070')


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


def forecast_XGBoost_request(
        col_target: str,
        last_know_index: int,
        evaluation_index: int,
        lag: int,
        model_architecture_params: list,
        json_list_df_all_data_norm: list,
        type: str,
        norm_values: bool
):
    url = f'{url_backend}/backend/v1/forecast_XGBoost'
    json = {
        "col_target": col_target,
        "last_know_index": last_know_index,
        "evaluation_index": evaluation_index,
        "lag": lag,
        "model_architecture_params": model_architecture_params,
        "json_list_df_all_data_norm": json_list_df_all_data_norm,
        "type": type,
        "norm_values": str(norm_values)
    }
    try:
        req = requests.post(
            url=url,
            json=json,
        )
        req_json = req.json()

        if req.status_code == 200:

            df_evaluetion_json = req_json['df_evaluetion']
            df_true_all_col_json = req_json['df_true_all_col']
            loss_list = req_json['loss_list']
            df_real_predict_json = req_json['df_real_predict']
            response_massage = req_json['response_massage']
            response_code = req_json['response_code']

            df_evaluetion = pd.DataFrame.from_dict(df_evaluetion_json)
            df_true_all_col = pd.DataFrame.from_dict(df_true_all_col_json)
            df_real_predict = pd.DataFrame.from_dict(df_real_predict_json)

            return df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_massage
        else:
            logger.error(f'Status code backend server: {req.status_code}')
            logger.error(req.status_code)
            return None, None, None

    except Exception as e:
        print('------------------------------------ ERROR ------------------------------------')
        print(e)


def xgb_predict(count_time_points_predict):

    model_architecture_params = {
        "objective": "reg:squarederror",
        "n_estimators": 500,
        "learning_rate": 0.1,
        "max_depth": 15,
        "subsample": .9,
        "colsample_bytree": .9,
        "min_child_weight": 5,
        "booster": "gbtree"

    }

    try:
        logger.info(f"Connecting to database with limit={lag}")
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        select_query = f"""
                SELECT * FROM {table_name} ORDER BY datetime DESC;
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
        print('is working 1')
        df_general_norm_df, min_val, max_val = normalization_request(
            col_time='datetime',
            col_target=measurement,
            json_list_df=json_list_general_norm_df
        )
        print('is working 2')

        df_general_norm_df = df_general_norm_df.replace("None", None)

        df_general_norm_df = df_general_norm_df.drop(columns=['datetime'])

        json_list_df_all_data_norm = df_general_norm_df.to_dict(orient='records')

        last_know_index = len(df_general_norm_df) - (count_time_points_predict + 1)
        evaluation_index = count_time_points_predict

        model_architecture_params = json.dumps([model_architecture_params])
        model_architecture_params = json.loads(model_architecture_params)
        print('is working 3')

        df_evaluation, df_true_all_col, loss_list, df_real_predict, \
            response_code, response_massage = forecast_XGBoost_request(
            col_target=measurement,
            last_know_index=last_know_index,
            evaluation_index=evaluation_index,
            lag=lag,
            model_architecture_params=model_architecture_params,
            json_list_df_all_data_norm=json_list_df_all_data_norm,
            type='predictions',
            norm_values=True
        )
        print('is working 4')


        json_list_df_real_predict = df_real_predict.to_dict(orient='records')

        logger.info("Reversing normalization on predictions.")

        print('is working 5')

        df_predict = reverse_normalization_request(
            col_time='datetime',
            col_target=measurement,
            json_list_norm_df=json_list_df_real_predict,
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
