import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from pathlib import Path
import pandas as pd

@st.cache_data
def load_data():
    data = pd.read_csv('https://dermdetect.s3.eu-west-3.amazonaws.com/metadata.csv')

    return data

def get_uploaded_image():
    upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if upload is not None:
        Path("data").mkdir(parents=True, exist_ok=True)
        # Create a directory and save the image file before proceeding. 
        file_path = os.path.join("data/", upload.name)
        with open(file_path, "wb") as user_file:
            user_file.write(upload.getbuffer())

        return file_path # fixed indentation
    
def predict(user_image):
    img = image.load_img(user_image, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    model = tf.keras.models.load_model("./model/resnet50_model.h5")
    result = model.predict(img_array)
    
    return result
    

option_sex = st.selectbox(
    "Choose your sex :",
    ("male", "female"),
    index = None
)

input_age = st.number_input("Insert your age :", min_value = 0, step=1, value = None)

data = load_data()

option_anatomie = st.selectbox(
    "Choose the location of the image :",
    data['anatom_site_general'].unique().tolist(),
    index = None
)

# File uploader
st.header('Upload an image of skin to determine if there is cancer.')

user_image = get_uploaded_image()

if st.button("Validate", type="primary"):
    if user_image is not None and option_sex is not None and input_age is not None and  option_anatomie is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.header("Image")
            st.image(user_image)
            st.write("Age: ", input_age)
            st.write("Sex: ", option_sex)
            st.write("Anatomie: ", option_anatomie)

        with col2:
            result = predict(user_image)
            st.header("Resnet50")
            st.write(f"Prediction: {result}")
    else:
        st.warning("Please upload an image and complete the form before proceeding.")
