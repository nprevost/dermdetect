import os
import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv(dotenv_path='../.env')

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("APP_URI_MLFLOW")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("InceptionV3_Metadata_Nico")
mlflow.tensorflow.autolog(log_models=True)

# Load dataset
csv_path = os.getenv('CSV_PATH')
df = pd.read_csv(csv_path + '/merge_metadata.csv')

# Convert `sex` to numerical values
df["sex"] = df["sex"].str.lower().map({"female": 1, "male": 0})

# Normalize `age_approx`
df["age_approx"] = df["age_approx"] / df["age_approx"].max()

# Convert `target` from string to integer (0 = benign, 1 = malignant)
df["target"] = df["target"].str.lower().map({"benign": 0, "malignant": 1})

# Sample 20% of dataset
df_sampled = df.sample(frac=0.2, random_state=42).reset_index(drop=True)

# Split into train (70%) and validation (30%)
train_df, val_df = train_test_split(df_sampled, test_size=0.3, random_state=42, stratify=df_sampled["target"])

# Image directory
IMAGE_DIR = os.getenv('IMAGE_DIR')

# 🔹 Custom Data Generator for Multi-Input Model
class MultiInputDataGenerator(Sequence):
    def __init__(self, dataframe, image_dir, batch_size=32, target_size=(299, 299), shuffle=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataframe))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.dataframe.iloc[batch_indexes]

        # Convert images
        images = np.stack([
            tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(
                    os.path.join(self.image_dir, img_id), target_size=self.target_size
                )
            ) / 255.0 for img_id in batch_data["image_id"]
        ]).astype(np.float32)

        # Convert metadata (sex, age)
        metadata = np.stack(batch_data[["sex", "age_approx"]].values).astype(np.float32)

        # Convert labels
        labels = np.array(batch_data["target"].values, dtype=np.float32)

        # Return a dictionary matching the model inputs
        return ({"image_input": tf.convert_to_tensor(images), 
                 "metadata_input": tf.convert_to_tensor(metadata)}, 
                tf.convert_to_tensor(labels))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Create data generators
train_generator = MultiInputDataGenerator(train_df, IMAGE_DIR, shuffle=True)
val_generator = MultiInputDataGenerator(val_df, IMAGE_DIR, shuffle=False)

# 🔹 InceptionV3 Model for Image Processing
image_input = Input(shape=(299, 299, 3), name="image_input")
base_model = InceptionV3(weights="imagenet", include_top=False, input_tensor=image_input)

# Unfreeze the last 30 layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:  
    layer.trainable = False

# Extract image features
x = GlobalAveragePooling2D()(base_model.output)

# 🔹 Additional Input for Metadata (Sex, Age)
metadata_input = Input(shape=(2,), name="metadata_input")  

# 🔹 Merge image and metadata features
merged = Concatenate()([x, metadata_input])

# Fully connected layers
x = Dense(256, activation="relu")(merged)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)  

# 🔹 Create Multi-Input Model
model = Model(inputs=[image_input, metadata_input], outputs=output)

# ✅ Compile with additional evaluation metrics
model.compile(optimizer=Adam(learning_rate=0.00001), 
              loss="binary_crossentropy", 
              metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

# ✅ Train model
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("train_size", len(train_df))
    mlflow.log_param("val_size", len(val_df))
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 1)
    mlflow.log_param("fine_tune_layers", 30)

    # ✅ Train model (fine-tuning from the start with metadata)
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=1
    )

    # ✅ Evaluate after training and log new metrics
    val_loss, val_acc, val_precision, val_recall, val_auc = model.evaluate(val_generator)
    
    # Log evaluation metrics to MLflow
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("val_accuracy", val_acc)
    mlflow.log_metric("val_precision", val_precision)
    mlflow.log_metric("val_recall", val_recall)
    mlflow.log_metric("val_auc", val_auc)

    # Évaluation sur l'ensemble de validation
    #y_true = val_generator.classes
    #y_pred_prob = model.predict(val_generator).ravel()
    #y_pred = (y_pred_prob > 0.5).astype("int32")

    # Tracé et enregistrement de la matrice de confusion
    #cm = confusion_matrix(y_true, y_pred)
    #df_cm = pd.DataFrame(cm, index=['Bénin', 'Malin'], columns=['Bénin', 'Malin'])
    #fig = px.imshow(df_cm, text_auto=True, color_continuous_scale='Blues')
    #fig.write_image("benchmark_models/artefacts/resnet50_confusion_matrix2.png", format="png", engine='kaleido')
    #mlflow.log_artifact("benchmark_models/artefacts/resnet50_confusion_matrix2.png")

    print("Fine-tuning completed and logged to MLflow!")

    # Save final fine-tuned model
    model.save("benchmark_models/artefacts/inceptionv3_skin_cancer_finetuned2.keras")
    mlflow.log_artifact("benchmark_models/artefacts/inceptionv3_skin_cancer_finetuned2.keras")
