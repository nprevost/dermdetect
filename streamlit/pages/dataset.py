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
                 names='sex')
    
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

def analyze_anatomie(df):
    df_benign_malignant_anatom = df.groupby('benign_malignant')['anatom_site_general'].value_counts()
    df_benign_malignant_anatom = df_benign_malignant_anatom.reset_index()
    df_benign_malignant_anatom.columns = ['benign_malignant', 'anatom_site_general', 'count']

    fig = px.bar(df_benign_malignant_anatom,
                    y='count',
                    x='benign_malignant',
                    color='anatom_site_general',
                    title='Repartition anatomie')
    
    st.plotly_chart(fig, theme=None)

def analyze_age_cancer(df):
    new_df = df.groupby('benign_malignant')['age_approx'].value_counts()
    new_df = new_df.reset_index()
    new_df.columns = ['benign_malignant', 'age', 'count']

    fig = px.bar(new_df,
                y='count',
                x='age',
                color='benign_malignant',
                barmode='group')
    
    st.plotly_chart(fig, theme=None)

def analyze_sex_cancer(df):
    new_df = df.groupby('benign_malignant')['sex'].value_counts()
    new_df = new_df.reset_index()
    new_df.columns = ['benign_malignant', 'sex', 'count']

    fig = px.bar(new_df,
                y='count',
                x='sex',
                color='benign_malignant',
                barmode='group')
    
    st.plotly_chart(fig, theme=None)

data = load_data()

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.dataframe(data)

analyze_type(data)

col1, col2 = st.columns(2)

with col1:
    st.header("Répartition H / F")
    analyze_sex(data)

with col2:
    st.header("Répartition du cancer H / F")
    analyze_sex_cancer(data)

st.header("Répartition du cancer par age")
analyze_age_cancer(data)

analyze_anatomie(data)
