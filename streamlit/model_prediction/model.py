import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from pathlib import Path

def get_uploaded_image():
    upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if upload is not None:
        Path("data").mkdir(parents=True, exist_ok=True)
        # Create a directory and save the image file before proceeding. 
        file_path = os.path.join("data/", upload.name)
        with open(file_path, "wb") as user_file:
            user_file.write(upload.getbuffer())

        return file_path # fixed indentation
    
def predict_with_resnet50(user_image):
    img = image.load_img(user_image, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    model = tf.keras.models.load_model("./model/resnet50_model.h5")
    result = model.predict(img_array)
    
    return result

def predict_with_vgg16(user_image):
    img = image.load_img(user_image, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    model = tf.keras.models.load_model("./model/VGG16_model.h5")
    result = model.predict(img_array)
    
    return result
    

# File uploader
st.header('Upload an image of skin to determine if there is cancer.')

user_image = get_uploaded_image()

if user_image is not None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Image")
        st.image(user_image)

    with col2:
        result = predict_with_resnet50(user_image)
        st.header("Resnet50")
        st.write(f"Prediction: {result}")

    with col3:
        result = predict_with_vgg16(user_image)
        st.header("VGG16")
        st.write(f"Prediction: {result}")
