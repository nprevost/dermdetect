import numpy as np
from tensorflow.keras.preprocessing import image

# Function to preprocess and predict a single image
def predict_image(model, img_path, img_size=(299, 299)):
    """
    Load an image, preprocess it, and make a prediction.
    
    Parameters:
    - model: Trained InceptionV3 model
    - img_path: Path to the image file
    - img_size: Target size for resizing (default 299x299 for InceptionV3)
    
    Returns:
    - Probability of malignant class (1 = malignant, 0 = benign)
    """
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image (same as training)

    # Make prediction
    prediction = model.predict(img_array)[0][0]  # Get probability

    # Interpret result
    if prediction >= 0.5:
        return f"Prediction: Malignant (Probability: {prediction:.2f})"
    else:
        return f"Prediction: Benign (Probability: {1 - prediction:.2f})"

# Example usage
img_path = "/Users/maurice/Documents/data_nogit/Dermdetect/SAMPLE_20/example_image.jpg"
result = predict_image(model, img_path)
print(result)