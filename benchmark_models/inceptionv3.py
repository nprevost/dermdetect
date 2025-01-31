import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ✅ Load environment variables
dotenv_path = r"D:\Desktop\dermproject\dermdetect\.env"
load_dotenv(dotenv_path)

# ✅ Get sample dataset paths from .env

SAMPLE_20_PATH = os.getenv("SAMPLE_20_PATH")  # Path to sample images
SAMPLE_20CSV_PATH = os.getenv("SAMPLE_20CSV_PATH")  # Path to sample dataset CSV


# ✅ Model parameters
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

# ✅ Configure MLflow
MLFLOW_URI = os.getenv("APP_URI_MLFLOW")
mlflow.set_tracking_uri(MLFLOW_URI)

EXPERIMENT_NAME = "inceptionV3L"
mlflow.set_experiment(EXPERIMENT_NAME)

mlflow.tensorflow.autolog()

# ✅ Load sample dataset CSV
df = pd.read_csv(SAMPLE_20CSV_PATH)

# ✅ Split dataset into training (70%) and validation (30%)
train_df, val_df = train_test_split(df, test_size=0.3, stratify=df['target'], random_state=42)

# ✅ Data Augmentation & Image Preprocessing
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=SAMPLE_20_PATH,  # ✅ Use sample dataset
    x_col="image_id",
    y_col="target",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='binary',
    seed=42,
    workers=1
)

val_generator = datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=SAMPLE_20_PATH,  # ✅ Use sample dataset
    x_col="image_id",
    y_col="target",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,  # No need to shuffle validation set
    class_mode='binary',
    seed=42,
    workers=1
)

# ✅ Load InceptionV3 as Base Model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False  # Freeze pretrained layers

# ✅ Custom Classification Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(1, activation='sigmoid')(x)

# ✅ Define Model
model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy', Precision(), Recall(), AUC()])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ✅ Start MLflow Tracking
with mlflow.start_run():
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("learning_rate", LEARNING_RATE)

    # ✅ Train Model
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator),
        callbacks=[early_stopping],
        verbose=1
    )

    # ✅ Evaluate Model & Log Metrics
    val_loss, val_accuracy, val_precision, val_recall, val_auc = model.evaluate(val_generator)
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("val_accuracy", val_accuracy)
    mlflow.log_metric("val_precision", val_precision)
    mlflow.log_metric("val_recall", val_recall)
    mlflow.log_metric("val_auc", val_auc)

    # ✅ Generate Predictions for Confusion Matrix
    y_true = val_generator.classes  # True labels
    y_pred_probs = model.predict(val_generator)  # Get probabilities
    y_pred = (y_pred_probs >= 0.5).astype(int).flatten()  # Convert to binary predictions

    # ✅ Compute Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", conf_matrix)

    # ✅ Save Confusion Matrix as Image
    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("benchmark_models/artefacts/confusion_matrix.png")
    plt.close()

    # ✅ Log Confusion Matrix Image to MLflow
    mlflow.log_artifact("benchmark_models/artefacts/confusion_matrix.png")

    # ✅ Save Classification Report
    class_report = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"])
    with open("benchmark_models/artefacts/classification_report.txt", "w") as f:
        f.write(class_report)

    # ✅ Log Classification Report to MLflow
    mlflow.log_artifact("benchmark_models/artefacts/classification_report.txt")

    # ✅ Save the model in the new Keras format
    model.save("benchmark_models/artefacts/inceptionv3_model.keras")  # Saves as .keras instead of .h5

    # ✅ Log the Keras model explicitly in MLflow
    mlflow.tensorflow.log_model(tf.keras.models.load_model("benchmark_models/artefacts/inceptionv3_model.keras"), "inceptionv3_model")


  


print("🎯 Training complete! Model logged to MLflow.")
