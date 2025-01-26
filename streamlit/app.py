import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import os

@st.cache_data
def load_data():
    data = pd.read_csv('./csv/dataset.csv')

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

    target_percentages['target'] = target_percentages['target'].map({1: 'Malignant', 0: 'Benign'})

    # Créer le diagramme circulaire avec Plotly
    fig = px.pie(target_percentages, values='percentage', names='target', title='Pourcentage de la colonne target')
    st.plotly_chart(fig, theme=None)