import requests

base_url = "http://0.0.0.0:7070/model_fast_api/v1"


def call_predict_api(count_time_points_predict: int):
    url = f"{base_url}/predict"
    payload = {"count_time_points_predict": count_time_points_predict}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")


def call_update_model_api():
    url = f"{base_url}/update_model"

    try:
        response = requests.post(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

# predict_response = call_predict_api(count_time_points_predict=288)
# print(predict_response)

# update_model_response = call_update_model_api()
# print(update_model_response)