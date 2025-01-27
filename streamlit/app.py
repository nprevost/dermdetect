import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import os
import numpy as np
from PIL import Image

@st.cache_data
def load_data():
    data = pd.read_csv('./csv/dataset.csv')

    data = data[~data['image_id'].str.endswith('_1.JPG')]
    data = data[~data['image_id'].str.endswith('_2.JPG')]
    data = data[~data['image_id'].str.endswith('_3.JPG')]
    data = data[~data['image_id'].str.endswith('_4.JPG')]

    return data

if __name__ == '__main__':
    st.set_page_config(
        page_title="Dermdetect App",
        layout="wide"
    )

    st.title("Dermdetect - Détection de cancer de la peau")

    with st.sidebar:
        selected = option_menu("Main Menu", ["Home", 'Settings'], 
            icons=['house', 'gear'], menu_icon="cast", default_index=1)
        selected


    data = load_data()

    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.dataframe(data)

    target_counts = data['target'].value_counts(normalize=True) * 100

    # Convertir les pourcentages en DataFrame pour Plotly
    target_percentages = target_counts.reset_index()
    target_percentages.columns = ['target', 'percentage']

    # Arrondir les pourcentages à 3 décimales et les convertir en chaîne de caractères
    target_percentages['percentage'] = target_percentages['percentage'].round(3).astype(str)

    # Créer le diagramme circulaire avec Plotly
    fig = px.pie(target_percentages,
                 values='percentage',
                 names='target',
                 title='Pourcentage de la colonne target')
    st.plotly_chart(fig, theme=None)

    # Créer le bar chart avec Plotly
    fig = px.bar(target_percentages,
                 x = 'target',
                 y='percentage',
                 color="percentage",
                 title='Pourcentage de la colonne target')
    st.plotly_chart(fig, theme=None)


    # File uploader
    st.header('Upload an image of skin to determine if there is cancer.')

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    image = None
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        st.image(image, caption='Uploaded Image.', use_container_width =True)

        # Créer un slider pour sélectionner la tranche
        slice_index = st.slider("Sélectionnez une tranche", 0, image.shape[0] - 1, 0)

        # Sélectionner la tranche
        slice_2d = image[slice_index]

        # Convertir la tranche en DataFrame
        df = pd.DataFrame(slice_2d, columns=[f'Col{i}' for i in range(slice_2d.shape[1])])

        # Afficher la tranche sous forme de DataFrame
        st.write(f"Tranche {slice_index} du tableau NumPy 3D:")
        st.dataframe(df)