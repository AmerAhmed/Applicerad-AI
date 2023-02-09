import os
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import load_model
from fastapi import FastAPI
from fastapi import UploadFile, File
from api.prediction import prediction1
import uvicorn

app = FastAPI()


# import models
model = tf.keras.models.load_model(os.path.abspath('model3test'))


@app.post('/api/upload')
# Request upload file returns predictions
async def UploadImage(file: bytes = File(...)):
    with open('pictures/prediction/image.jpg', 'wb') as image:
        image.write(file)
        image.close()
    prediction = prediction1(model)
    return prediction
