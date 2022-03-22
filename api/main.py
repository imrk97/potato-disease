from fastapi import FastAPI, UploadFile, File
from enum import Enum
import uvicorn
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

#model = tf.keras.models.load_model('../models/2')
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/hello/{name}')
async def hello(name):
    return 'hello! welcome to fastapi tutorials {}'.format(name)


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

classes = ['Early_blight', 'Late_blight', 'Healthy']

def load_model():
    return tf.keras.models.load_model('../models/2')
@app.post('/predict')
async def predict(
        file: UploadFile = File(...)
) -> object:
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)
    model = load_model()
    print('Model Loaded: ', model.summary())
    prediction = model.predict(image_batch)
    ans = classes[np.argmax(prediction[0])]
    print(ans)
    predicted_class = classes[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


@app.get('/ping')
async def ping():
    return 'The website is alive.'


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8080)
