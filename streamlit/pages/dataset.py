import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Dataset")

@st.cache_data
def load_data():
    data = pd.read_csv('https://dermdetect.s3.eu-west-3.amazonaws.com/metadata.csv')

    return data

def analyze_sex(df):
    df_sex = df['sex'].value_counts()
    df_sex = df_sex.reset_index()

    # Renommer les colonnes
    df_sex.columns = ['sex', 'count']

    fig = px.pie(df_sex,
                 values='count',
                 names='sex',
                 title='Répartition H / F')
    
    st.plotly_chart(fig, theme=None)

def analyze_type(df):
    df_diagnosis = df['benign_malignant'].value_counts()
    df_diagnosis = df_diagnosis.reset_index()

    # Renommer les colonnes
    df_diagnosis.columns = ['benign_malignant', 'count']

    fig = px.pie(df_diagnosis,
                 values='count',
                 names='benign_malignant',
                 title='Repartition Benign / Malignant')

    st.plotly_chart(fig, theme=None)

def analyze_age(df):
    df_age = df['age_approx'].value_counts()
    df_age = df_age.reset_index()

    # Renommer les colonnes
    df_age.columns = ['age', 'count']

    fig = px.bar(df_age,
                 y='count',
                 x='age',
                 title='Repartition Age')

    st.plotly_chart(fig, theme=None)

data = load_data()

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.dataframe(data)

col1, col2 = st.columns(2)

with col1:
    st.header("Sexe")
    analyze_sex(data)

with col2:
    st.header("Benin ou Maligne")
    analyze_type(data)

analyze_age(data)
