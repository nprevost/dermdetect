import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 About the Dataset")
st.subheader("🔬 Skin Cancer Dataset from ISIC")

st.markdown("""
This application is powered by a dataset from **ISIC: The International Skin Imaging Collaboration**, a globally recognized initiative for skin imaging research.

### 📌 Dataset Overview:
- **Total Images:** **44,164**  
- **Benign Cases:** **37,055** (84%)  
- **Malignant Cases:** **7,109** (16%)  
- **Additional Features:** **Patient Sex & Age**  

Our dataset was carefully curated to ensure **a balance between medical relevance and real-world application**, making it highly effective for AI-driven skin cancer detection.

### 🏥 Key Features Used in the Model:
Besides image data, our model also considers:
- **Sex** of the patient (Male/Female)
- **Age** of the patient (to assess risk factors)

By incorporating **sex and age**, we aim to improve the accuracy of predictions and provide a more **personalized risk assessment**.

### 🔍 Sample Data:
""")

@st.cache_data
def load_data():
    url_csv = 'https://dermdetect.s3.eu-west-3.amazonaws.com/merge_metadata.csv'
    data = pd.read_csv(url_csv)

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
    st.header("Distribution male / female")
    analyze_sex(data)

with col2:
    st.header("Distribution of cancer male / female")
    analyze_sex_cancer(data)

st.header("Distribution of cancer by age")
analyze_age_cancer(data)

analyze_anatomie(data)

# Add a button to go to upload page
if st.button("Start Now 🚀"):
    st.switch_page("./pages/model.py")