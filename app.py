import streamlit as st
import tensorflow as tf 
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
import causes_remedies 
MODEL = tf.keras.models.load_model('D:\Data Science\Major project\Skin Disease\model.h5')

classes = ['cellulitis','impetigo','athlete-foot','nail-fungus',
           'ringworm','cutaneous-larva-migrans','chickenpox','shingles']


disease_type = {
    'cellulitis': 'Bacterial Infection',
    'impetigo': 'Bacterial Infection',
    'athlete-foot': 'Fungal Infection',
    'nail-fungus': 'Fungal Infection',
    'ringworm': 'Fungal Infection',
    'cutaneous-larva-migrans': 'Parasitic Infection',
    'chickenpox': 'Viral Infection',
    'shingles': 'Viral Infection',
}

def read_image(file):
    img = Image.open(BytesIO(file))
    img = tf.keras.preprocessing.image.img_to_array(img)
    return img

st.header("Skin Disease Detection Web-App")
st.write('Upload image of infected area to predict the disease',)
uploaded_img = st.file_uploader("upload image here",type=['jpeg','jpg','png'])

button = st.button("Predict")

if uploaded_img:
    if button:
        st.write("processing......")

        img_bytes =  uploaded_img.read() 
        img = read_image(img_bytes)
        img = tf.image.resize(img, (224, 224))
        img_batch = tf.expand_dims(img,0)

        prediction = MODEL.predict(img_batch)
        pred_class = classes[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])*100
        type_of_disease = disease_type.get(pred_class)

        probabilities = np.array(prediction[0])


        st.subheader(f'Predicted Disease: {pred_class}')
        st.write(f'Confidence on prediction {confidence: .2f}%')
        st.write(f'Type of Disease: {type_of_disease}')

        st.write("Finding Causes and Remedies.......")
        
        cr = causes_remedies.call_causes(pred_class)
        with st.expander(f'Causes and Remedies for {pred_class}'):
            st.write(cr)


        