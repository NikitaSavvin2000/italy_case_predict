import uvicorn
from typing import Annotated, List
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

from src.config import logger, public_or_local
from src.models.schemes import PredictRequest
from src.utils.forecast import create_predict
from src.utils.learn_model import update_model


if public_or_local == 'LOCAL':
    url = 'http://localhost'
else:
    url = 'http://11.11.11.11'

origins = [
    url
]

# Docs - http://0.0.0.0:7070/model_fast_api/v1/
# Docs - http://77.37.136.11:7072/model_fast_api/v1/
app = FastAPI(docs_url="/model_fast_api/v1/", openapi_url='/model_fast_api/v1/openapi.json')
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/model_fast_api/v1/predict")
async def predict(body: Annotated[
    PredictRequest, Body(
        example={"count_time_points_predict": 288})]):
    try:
        count_time_points_predict = body.count_time_points_predict
        logger.info(f"Starting prediction with count_time_points_predict={count_time_points_predict}")
        create_predict(count_time_points_predict=count_time_points_predict)
        logger.info("Prediction process completed successfully.")
        return {"message": f"Prediction for {count_time_points_predict} time points completed successfully."}
    except Exception as ApplicationError:
        logger.error(f"Error occurred during prediction: {ApplicationError.__repr__()}")
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )


@app.post("/model_fast_api/v1/update_model")
async def update_model_api():
    try:
        logger.info("Starting model update process.")
        update_model()
        logger.info("Model update completed successfully.")
        return {"message": "Model updated successfully."}
    except Exception as ApplicationError:
        logger.error(f"Error occurred during model update: {ApplicationError.__repr__()}")
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )



@app.get("/")
def read_root():
    return {"message": "Welcome to the indicators System API"}


if __name__ == "__main__":
    port = 7070
    uvicorn.run(app, host="0.0.0.0", port=port)
