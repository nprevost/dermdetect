import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Dataset")

@st.cache_data
def load_data():
    data = pd.read_csv('./csv/dataset.csv')

    data = data[~data['image_id'].str.endswith('_1.JPG')]
    data = data[~data['image_id'].str.endswith('_2.JPG')]
    data = data[~data['image_id'].str.endswith('_3.JPG')]
    data = data[~data['image_id'].str.endswith('_4.JPG')]

    return data

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