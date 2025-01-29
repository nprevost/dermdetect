import streamlit as st

st.set_page_config(
    page_title="Dermdetect App",
    layout="wide"
)

st.header("Dermdetect - Détection de cancer de la peau")

intro_page = st.Page('intro.py', title="Intro", icon=":material/home:")
modele_page = st.Page('model_prediction/model.py', title="Prediction", icon=":material/batch_prediction:")
dataset_page = st.Page("dataset/app.py", title="Dataset", icon=":material/dataset:")

pages = [intro_page, dataset_page, modele_page]

page_dict = {}

page_dict["pages"] = pages

pg = st.navigation({"Menu": pages})

pg.run()