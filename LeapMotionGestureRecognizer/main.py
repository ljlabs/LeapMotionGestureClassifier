# main.py
import time

import uvicorn
from fastapi import FastAPI

from Handlers.Recognizer import Recognizer
from Models.InferenceResponse import InferenceResponse
from Models.inferenceRequest import InferenceRequest

app = FastAPI()
recognizer = Recognizer()


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    t0 = time.time()
    prediction = recognizer.infer(request.get_data)
    classification, confidence = recognizer.get_class(prediction)

    response = InferenceResponse.asJson(classification, confidence, prediction, time.time() - t0)
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
