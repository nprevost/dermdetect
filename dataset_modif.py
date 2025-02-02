import streamlit as st
import pandas as pd

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

# Load dataset (Ensure the correct path)
df = pd.read_csv("streamlit/data/merged_cleaned_dataset.csv")  # Adjust path if needed
st.write(df.head())  # Display first few rows

# Display class distribution
st.subheader("📊 Class Distribution")
class_counts = df["target"].value_counts().rename(index={0: "Benign", 1: "Malignant"})
st.bar_chart(class_counts)

st.markdown("""
📌 *Navigate through the sidebar to explore model predictions!*
""")
