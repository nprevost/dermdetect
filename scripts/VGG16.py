import os
import shutil
import mlflow
import mlflow.tensorflow 
import numpy as np
import pandas as pd
import plotly
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.metrics import auc, confusion_matrix
import plotly.express as px


# Load environment variables from .env file
load_dotenv(dotenv_path=r"C:\\Users\\Lian4ik\\Desktop\\cancer_project\\dermdetect\\.env")  # Adjust the path to your .env file

# AWS Credentials
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("APP_URI_MLFLOW")  # Your MLflow server URI
EXPERIMENT_NAME = "feat_VGG16"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# Access paths 
CSV_PATH = r"C:\Users\Lian4ik\Desktop\cancer_project\data\dataset.csv"
IMAGE_DIR = r"C:\Users\Lian4ik\Desktop\cancer_project\data\ALL_IMAGES"

SAMPLE_IMAGE_DIR= r"C:\Users\Lian4ik\Desktop\cancer_project\data\sample_images"
SAMPLE_CSV_PATH= r"C:\Users\Lian4ik\Desktop\cancer_project\data\sample_dataset.csv"



# Step 1: Load and Split the Dataset
# Load the sampled dataset
df_sample = pd.read_csv(SAMPLE_CSV_PATH)

# Ensure target column values are strings
df_sample["target"] = df_sample["target"].astype(str)

#Preprocessing
img_generator = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.3
)

#Preprocessing for train set
train_generator = img_generator.flow_from_dataframe(
    dataframe=df_sample,
    directory=SAMPLE_IMAGE_DIR,
    x_col="image_id",
    y_col="target",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=True,
    subset="training",
    seed=42
)

#Preprocessing for val set
val_generator = img_generator.flow_from_dataframe(
    dataframe=df_sample,
    directory=SAMPLE_IMAGE_DIR,
    x_col="image_id",
    y_col="target",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=True,
    subset="validation",
    seed=42
)


# Enable MLflow TensorFlow -> Keras autologging
mlflow.keras.autolog()


# Dynamically calculate steps per epoch
steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)

# Step 3: Define the VGG16 Model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

# Add custom layers
x = Flatten()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)  # Binary classification

# Build the model
model = Model(inputs=base_model.input, outputs=output)

LR = 0.001
EPOCHS = 1

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(name="auc")]
)


# Step 4: Train the Model with MLflow Tracking
with mlflow.start_run():
    # Log model parameters
    mlflow.log_param("model", "VGG16")
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("dropout_rate", 0.5)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("train_val_split", 0.7)



    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=1,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    # model's performance on the validation dataset
    y_true = val_generator.classes
    y_pred_prob = model.predict(val_generator).ravel()
    y_pred = (y_pred_prob > 0.5).astype("int32")

    # Calcul des métriques de classification
    fpr, tpr, thresholds = roc_curve (y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    accuracy = history.history['val_accuracy'][-1]

    mlflow.log_metric("Validation AUC", roc_auc)
    mlflow.log_metric("Validation Accuracy", accuracy)

    # Sauvegarde des courbes d'apprentissage
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Courbe de Perte")
    plt.xlabel("Époques")
    plt.ylabel("Loss")
    plt.savefig("loss_plot.png")
    mlflow.log_artifact("loss_plot.png")


    # Tracé et enregistrement de la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=['Bénin', 'Malin'], columns=['Bénin', 'Malin'])
    fig = px.imshow(df_cm, text_auto=True, color_continuous_scale='Blues')
    fig.write_image("confusion_matrix.png", format="png", engine='kaleido')
    mlflow.log_artifact("confusion_matrix.png")

    # Enregistrement du modèle
    model.save("VGG16_model.h5")
    mlflow.log_artifact("VGG16_model.h5")

    # Enregistrement du seuil optimal basé sur la courbe ROC
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    mlflow.log_param("Optimal Threshold", optimal_threshold)

    print(f"Seuil optimal basé sur le point de coude : {optimal_threshold:.4f}")

print("Enregistrement des métriques et du modèle terminé.")

