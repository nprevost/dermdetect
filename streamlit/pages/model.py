import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import config
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
    
def predict(user_image, sex, age):
    img = image.load_img(user_image, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    sex_numeric = 1 if sex.lower() == "female" else 0

    age_max = 85
    age_normalized = (round(age / 5) * 5) / age_max

    # Metadonnées formatées pour TensorFlow
    metadata = np.array([[sex_numeric, age_normalized]], dtype=np.float32)

    config.enable_unsafe_deserialization()

    model = tf.keras.models.load_model("./model/model.keras")
    result = model.predict({"image_input": img_array, "metadata_input": metadata})
    
    return result
    

option_sex = st.selectbox(
    "Choose your sex :",
    ("male", "female"),
    index = None
)

input_age = st.number_input("Insert your age :", min_value = 0, step=1, value = None)

# File uploader
st.header('Upload an image of skin to determine if there is cancer.')

user_image = get_uploaded_image()

if st.button("Validate", type="primary"):
    if user_image is not None and option_sex is not None and input_age is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.header("Image")
            st.image(user_image)
            st.write("Age: ", input_age)
            st.write("Sex: ", option_sex)

        with col2:
            result = predict(user_image, option_sex, input_age)
            st.header("Model")

            predict = result[0][0] * 100

            st.write(f"Prediction: {predict:.2f}% of having cancer")
    else:
        st.warning("Please upload an image and complete the form before proceeding.")
