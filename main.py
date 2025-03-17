from fastapi import FastAPI,File,UploadFile
import uvicorn
import tensorflow as tf 
from PIL import Image
from io import BytesIO
import numpy as np

app = FastAPI()

MODEL = tf.keras.models.load_model('model.h5')
classes = ['cellulitis','impetigo','athlete-foot','nail-fungus',
           'ringworm','cutaneous-larva-migrans','chickenpox','shingles']
@app.get('/')
async def read_root():
    return {"Hello": "World"}

async def read_image(file):
    img = Image.open(BytesIO(file))
    img = tf.keras.preprocessing.image.img_to_array(img)
    return img

@app.post('/predict')
async def predict_image(file: UploadFile = File(...)):
    img_bytes = await file.read() 
    img = await read_image(img_bytes)
    img = tf.image.resize(img, (224, 224))
    img_batch = tf.expand_dims(img,0)
    
    prediction = MODEL.predict(img_batch)
    pred_class = classes[np.argmax(prediction[0])]
    return { 'class': pred_class,
             'confidence': str(np.max(prediction[0])), 
             'probability': prediction[0].tolist()
            }


if __name__ == '__main__':
    uvicorn.run(app,host='localhost',port=8000)