import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()
model = joblib.load('C:\\Users\\mitsa\\PycharmProjects\\python_JN\\model\\final_ds.pkl')


class Form(BaseModel):
    session_id: str
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    device_category: str
    device_brand: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    session_id: str
    results: float


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'session_id': form.session_id,
        'results': y[0]
    }
