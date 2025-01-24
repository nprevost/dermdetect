import os
import shutil
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("../dermdetect/.env")

# Access the AWS credentials
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

# Set the environment variables
os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

# Paths
base_dir = "../data/raw"  # Original dataset directory with BEN and MAL
sampled_dir = "../data/sample"  # Directory for sampled data
train_dir = os.path.join(sampled_dir, "train")
val_dir = os.path.join(sampled_dir, "val")

# MLflow Configuration
mlflow.set_tracking_uri("https://nprevost-dermdetect-mlflow.hf.space")  # Replace with your MLflow server URL
mlflow.set_experiment("liana_1st_exp")  # Replace with your experiment name

# Step 1: Sample 20% of the Dataset
def create_sampled_dataset(base_dir, sampled_dir, sample_frac=0.2):
    if os.path.exists(sampled_dir):
        shutil.rmtree(sampled_dir)  # Clean up any previous runs
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Create lists of all images
    data = []
    for label, folder in enumerate(["BEN", "MAL"]):
        folder_path = os.path.join(base_dir, folder)
        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                data.append((os.path.join(folder_path, img_name), label))
    
    # Convert to DataFrame and sample
    data_df = pd.DataFrame(data, columns=["image_path", "label"])
    sampled_df = data_df.sample(frac=sample_frac, random_state=42)

    # Split sampled data into train and validation sets
    train_df, val_df = train_test_split(sampled_df, test_size=0.3, random_state=42, stratify=sampled_df["label"])

    # Copy sampled images to the new directories
    for subset, subset_dir in [(train_df, train_dir), (val_df, val_dir)]:
        for _, row in subset.iterrows():
            label_folder = "BEN" if row["label"] == 0 else "MAL"
            dest_folder = os.path.join(subset_dir, label_folder)
            os.makedirs(dest_folder, exist_ok=True)
            shutil.copy(row["image_path"], dest_folder)

# Create the sampled dataset
create_sampled_dataset(base_dir, sampled_dir)

# Step 2: Set Up Image Generators
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Dynamically calculate steps to prevent insufficient data issues
steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)

# Step 3: Set Up VGG16 Model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the model with additional metrics
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])

# Step 4: Train the Model with MLflow Tracking
with mlflow.start_run():
    # Log model parameters
    mlflow.log_param("model", "VGG16")
    mlflow.log_param("learning_rate", 0.0001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("dropout_rate", 0.5)
    mlflow.log_param("epochs", 10)
    mlflow.log_param("sample_size", 0.2)
    mlflow.log_param("test_split", 0.3)

    # Enable MLflow TensorFlow autologging
    mlflow.tensorflow.autolog()

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    # Save the trained model
    model_path = "../models/vgg16_sampled_model.h5"
    model.save(model_path)

    # Log the saved model as an artifact
    mlflow.log_artifact(model_path)

    # Evaluate the model on validation data
    val_loss, val_accuracy, val_precision, val_recall = model.evaluate(val_generator)

    # Log metrics
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("val_accuracy", val_accuracy)
    mlflow.log_metric("val_precision", val_precision)
    mlflow.log_metric("val_recall", val_recall)

    # Calculate and log the AUC score
    y_true = val_generator.classes  # True labels
    y_pred = model.predict(val_generator)  # Predicted probabilities
    auc = roc_auc_score(y_true, y_pred)
    mlflow.log_metric("val_auc", auc)

    print(f"AUC Score: {auc:.4f}")
