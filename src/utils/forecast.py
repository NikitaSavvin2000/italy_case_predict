import psycopg2
import numpy as np
import pandas as pd
import tensorflow as tf

from psycopg2.extras import execute_values
from tensorflow.keras.models import load_model


table_name = 'load_consumption'
measurement = 'load_consumption'

table_predict_name = 'predict_load_consumption'

'''"Эта часть задается из yaml конфига'''
lag =
points_per_call =
path_to_mpdel = "/Users/dmitrii/Downloads/model.h5"


model = load_model(path_to_mpdel)

DB_PARAMS = {
    "dbname": "mydb",
    "user": "myuser",
    "password": "mypassword",
    "host": "77.37.136.11",
    "port": 8083
}

limit = lag

conn = psycopg2.connect(**DB_PARAMS)
cur = conn.cursor()

select_query = f"""
SELECT * FROM {table_name} ORDER BY datetime DESC LIMIT {limit};
"""
cur.execute(select_query)
rows = cur.fetchall()

df_last_values = pd.DataFrame(rows, columns=["datetime", measurement])
df_last_values["datetime"] = df_last_values["datetime"].dt.tz_localize(None)



def create_x_input(df_to_predict, n_steps):
    df_input = df_to_predict.iloc[len(df_to_predict) - n_steps:]
    x_input = df_input.values
    return x_input


def make_predictions(x_input, x_future, points_per_call):
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


'''
Нужно создать df_predict который будет содержать значения для записи предсказания
Допустим наши последнее извстное значение это 17-02-2025 15:42:00 значит нужно сгенерировать df y на n-количество
 точек с интервалом 5 минут после даты 17-02-2025 15:42:00. То есть 17-02-2025 15:47:00 и тд 
'''
# код здесь


'''
Нужно отнормировать значения по нашей функции, она уже развернута на tool_backend и нужно написать обращение к ней
Пример обращения можно найти в репозитории https://github.com/NikitaSavvin2000/tool, а конкретно
Функция API вызова - https://github.com/NikitaSavvin2000/tool/blob/4ad5c623ffdeb597b1fe56dba1eb4e52842c6ccc/src/ui/api/api_calls.py#L12
Сам вызов - https://github.com/NikitaSavvin2000/tool/blob/4ad5c623ffdeb597b1fe56dba1eb4e52842c6ccc/src/ui/backend/forecast.py#L436
PS перед подачей на API нужно все переводить в json формат пример - https://github.com/NikitaSavvin2000/tool/blob/4ad5c623ffdeb597b1fe56dba1eb4e52842c6ccc/src/ui/backend/forecast.py#L434
'''

# код здесь
df_to_predict_norm =
df_predict_norm =

'''Далее готовим данные'''


x_input = create_x_input(df_to_predict_norm, lag)
x_future = df_predict_norm.values
predict_values = make_predictions(x_input, x_future, points_per_call)

df_predict_norm['P_l'] = predict_values


'''Это заменить на вызов API функции как и при нормализации аналогично сама функция выова + сам вызов найти можно +- там-же'''
df_predict = df_denormalize_with_meta(df_predict_norm, min_val, max_val)


 # код

"""Загрузка предсказания df_predict в базу данных"""


DB_PARAMS = {
    "dbname": "mydb",
    "user": "myuser",
    "password": "mypassword",
    "host": "77.37.136.11",
    "port": 8083
}


df_predict['datetime'] = pd.to_datetime(df_predict['datetime'])

data = [(row['datetime'], row['value']) for _, row in df_predict.iterrows()]


conn = psycopg2.connect(**DB_PARAMS)
cur = conn.cursor()

query = f"""
    INSERT INTO {table_predict_name} (datetime, {measurement})
    VALUES %s
    ON CONFLICT (datetime)
    DO UPDATE SET {measurement} = EXCLUDED.{measurement};
"""

execute_values(cur, query, data)

conn.commit()
cur.close()
conn.close()
