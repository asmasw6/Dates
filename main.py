from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from fastapi.middleware.cors import CORSMiddleware



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

MODEL = tf.keras.models.load_model("saved_models/2")
CLASS_NAMES = ['Ajwa', 'Galaxy', 'Medjool', 'Meneifi', 'Nabtat Ali', 'Rutab', 'Shaishe', 'Sokari', 'Sugaey']





@app.get("/ping")
async def ping():
    return "Hello T Date "


def readFileAsImage(data) -> np.ndarray:
    # data is byates
   return np.array(Image.open(BytesIO(data)).resize((200, 200))) 


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # bytes = await file.read()
    image = readFileAsImage(await file.read())
    img_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(img_batch) 
    
    predected_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = f"{np.max(prediction[0]) * 100:.2f}%"
    return {
        'class': predected_class,
        'confidence': confidence
        }
   


# if __name__=="__main__":
# uvicorn.run(app, port=8001, host='localhost')
