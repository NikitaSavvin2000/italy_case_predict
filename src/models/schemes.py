from pydantic import BaseModel


class PredictRequest(BaseModel):
    count_time_points_predict: int