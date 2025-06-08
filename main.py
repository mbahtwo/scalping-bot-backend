from fastapi import FastAPI
from pydantic import BaseModel
import requests
import tensorflow as tf
import numpy as np
import hmac
import hashlib
import time
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

class PriceRequest(BaseModel):
    prices: list[float]

class SignalResponse(BaseModel):
    signalText: str

interpreter = tf.lite.Interpreter(model_path="scalping_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
BASE_URL = "https://api.tokocrypto.com"

def auto_trade(symbol="BTCUSDT", side="BUY", quantity="0.0001"):
    path = "/api/v3/order"
    url = BASE_URL + path
    timestamp = int(time.time() * 1000)
    params = f"symbol={symbol}&side={side}&type=MARKET&quantity={quantity}&timestamp={timestamp}"
    signature = hmac.new(SECRET_KEY.encode(), params.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": API_KEY}
    full_url = f"{url}?{params}&signature={signature}"
    res = requests.post(full_url, headers=headers)
    return res.json()

def predict_signal(prices: list[float]) -> str:
    scaled = np.array(prices, dtype=np.float32).reshape(-1, 1)
    scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min() + 1e-8)
    input_data = scaled.reshape(1, 30, 1)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    signal = np.argmax(output)
    return ["SELL", "HOLD", "BUY"][signal]

@app.post("/predict", response_model=SignalResponse)
def get_signal(data: PriceRequest):
    signal = predict_signal(data.prices)
    return SignalResponse(signalText=signal)

@app.get("/live-signal")
def live_signal():
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=30"
    res = requests.get(url).json()
    prices = [float(k[4]) for k in res]
    signal = predict_signal(prices)
    result = {"signalText": signal}
    if signal in ["BUY", "SELL"]:
        result["trade"] = auto_trade(side=signal)
    return result
