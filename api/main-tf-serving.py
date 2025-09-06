from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import requests

# docker run -t --rm -p 8501:8501 -v C:\Users\HP\PycharmProjects\date_classfication:/date_classfication tensorflow/serving --rest_api_port=8501 --model_config_file=/date_classfication/models.config
app = FastAPI()

endpoint= 'http://localhost:8501/v1/models/date_classification_model:predict'
CLASS_NAMES = ['Ajwa', 'Galaxy', 'Medjool', 'Meneifi', 'Nabtat Ali', 'Rutab', 'Shaishe', 'Sokari', 'Sugaey']





@app.get("/ping")
async def ping():
    return "Hello T Date "


def readFileAsImage(data) -> np.ndarray:
    # data is byates
   return np.array(Image.open(BytesIO(data)).resize((200, 200))) 


@app.post("/predict")
async def predict(file: UploadFile = File(...)

):
    # bytes = await file.read()
    image = readFileAsImage(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    json_data={
        "instances": img_batch.tolist()
    }
    
    response= requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0]) # single image
    
    
    predected_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    
    return {
        'class': predected_class,
        'confidence': confidence
        }
   


if __name__=="__main__":
    uvicorn.run(app, port=8000, host='localhost')