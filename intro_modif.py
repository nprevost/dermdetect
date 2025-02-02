import streamlit as st

st.title("🩺 Skin Cancer Detection App")
st.subheader("🔍 AI-Powered Analysis for Early Detection")

st.markdown("""
Welcome to the **Skin Cancer Detection App**, where AI helps detect potential skin cancer conditions through image analysis.

### 🌟 How It Works:
1️⃣ **Upload a Dermoscopic Image** of a skin lesion.  
2️⃣ The AI model **analyzes the image** using deep learning.  
3️⃣ You receive a **prediction** indicating if the lesion is **benign or malignant**.  

### 🛠 Features:
- **Fast analysis with accuracy around 90%**
- **Uses deep learning model**
- **Simple and user-friendly interface**

### 📸 Important: Dermoscopic Images Required
To ensure accurate predictions, please upload a **dermoscopic image**.  
A dermoscopic image is a high-resolution **close-up** of a skin lesion taken with a **dermoscope**, a specialized magnifying tool used by dermatologists.  

#### 🔹 How to Take a Dermoscopic Image:
- **By a Doctor:** Dermatologists use a **professional dermoscope** to capture high-quality images.  
- **By Yourself:** You can use a **mobile dermoscope** attached to your smartphone to take a clear close-up of the lesion.  

📌 *Uploading non-dermoscopic images (e.g., normal photos taken without magnification) may result in inaccurate predictions.*

### 🔍 Example of a Mobile Dermoscope:
Here is an example of a **mobile dermoscope** that can be attached to a smartphone:

""")

# Display an image of a mobile dermoscope (replace with actual image path)
st.image("streamlit/assets/mobile_dermoscope.jpg", caption="Example of a Mobile Dermoscope", use_column_width=True)

st.markdown("""
👉 **Use the sidebar to navigate and upload an image to start the analysis.**
""")

# Add a button to go to the dataset or upload page
if st.button("Start Now 🚀"):
    st.switch_page("pages/dataset.py")  # Update this if necessary
