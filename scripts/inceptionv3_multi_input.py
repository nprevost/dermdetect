import os
import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.metrics import auc, confusion_matrix
import plotly.express as px


# Load environment variables from .env file
load_dotenv(dotenv_path= "../.env")  # Adjust the path to your .env file

# AWS Credentials
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("APP_URI_MLFLOW")  # Your MLflow server URI

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# 🔹 Set MLflow Experiment Name
mlflow.set_experiment("SkinCancer_Exp1_InceptionV3_Metadata")

mlflow.tensorflow.autolog(log_models=True)

# Load merged dataset
data_path = "../../data/merged_cleaned_dataset.csv" # Ensure using the cleaned dataset
df = pd.read_csv(data_path)

# ✅ Convert `sex` to numerical values (Ensure correct mapping)
df["sex"] = df["sex"].str.lower().map({"female": 1, "male": 0})

# Normalize `age` column (scale between 0 and 1)
df["age_approx"] = df["age_approx"] / df["age_approx"].max()

# 🔹 Use only 20% of images for training (random sample)
df_sampled = df.sample(frac=0.2, random_state=42).reset_index(drop=True)

# Split into train (70%) and validation (30%)
train_df, val_df = train_test_split(df_sampled, test_size=0.3, random_state=42, stratify=df_sampled["target"])

# Ensure `target` is string format for binary classification
train_df["target"] = train_df["target"].astype(str)
val_df["target"] = val_df["target"].astype(str)

# Image directory
IMAGE_DIR = "../../data/ALL_IMAGES"

# 🔹 Custom Data Generator for Multi-Input Model (Fixing TensorSpec Issue)
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

        # ✅ Convert images to proper TensorFlow format
        images = np.stack([
            tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(
                    os.path.join(self.image_dir, img_id), target_size=self.target_size
                )
            ) / 255.0 for img_id in batch_data["image_id"]
        ]).astype(np.float32)

        # ✅ Convert metadata (sex, age) to TensorFlow tensors
        metadata = np.stack(batch_data[["sex", "age_approx"]].values).astype(np.float32)

        # ✅ Convert labels to NumPy array
        labels = np.array(batch_data["target"].astype(float).values, dtype=np.float32)

        # 🔹 Wrap in TensorFlow tensors to avoid `TypeError`
        return (tf.convert_to_tensor(images), tf.convert_to_tensor(metadata)), tf.convert_to_tensor(labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# 🔹 Convert Generator to TensorFlow Dataset
def create_tf_dataset(generator):
    return tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(
            (tf.TensorSpec(shape=(None, 299, 299, 3), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 2), dtype=tf.float32)),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )


# 🔹 Convert Generator to TensorFlow Dataset (The from_generator() function expects a callable (function), not an instance of the generator.)
#def create_tf_dataset(generator_class, *args, **kwargs):
    #return tf.data.Dataset.from_generator(
     #   lambda: generator_class(*args, **kwargs),  # ✅ Proper callable function
      #  output_signature=(
       #     (tf.TensorSpec(shape=(None, 299, 299, 3), dtype=tf.float32),
        #     tf.TensorSpec(shape=(None, 2), dtype=tf.float32)),
         #   tf.TensorSpec(shape=(None,), dtype=tf.float32)
        #)
    #)




# 🔹 Create data generators
train_generator = MultiInputDataGenerator(train_df, IMAGE_DIR, shuffle=True)
val_generator = MultiInputDataGenerator(val_df, IMAGE_DIR, shuffle=False)

# 🔹 InceptionV3 Model for Image Processing
image_input = Input(shape=(299, 299, 3), name="image_input")
base_model = InceptionV3(weights="imagenet", include_top=False, input_tensor=image_input)

# ✅ Unfreeze the last 30 layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Freeze all but last 30 layers
    layer.trainable = False

# Extract image features
x = GlobalAveragePooling2D()(base_model.output)

# 🔹 Additional Input for Metadata (Sex, Age)
metadata_input = Input(shape=(2,), name="metadata_input")  # 2 features: sex & age

# 🔹 Merge image and metadata features
merged = Concatenate()([x, metadata_input])

# Fully connected layers
x = Dense(256, activation="relu")(merged)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)  # Binary classification

# 🔹 Create Multi-Input Model
model = Model(inputs=[image_input, metadata_input], outputs=output)

# ✅ Compile with additional evaluation metrics
model.compile(optimizer=Adam(learning_rate=0.00001), 
              loss="binary_crossentropy", 
              metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])



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

    print("Fine-tuning completed and logged to MLflow!")

    # 🔹 Log the trained model to MLflow
    mlflow.keras.log_model(model, "keras_model")



# 🔹 Save the model locally as `.keras`
os.makedirs("../models", exist_ok=True)  # ✅ Ensure directory exists
model.save("../models/inceptionv3_skin_cancer_finetuned.keras")  # ✅ Save in Keras format


print("I'm done!")


# - accuracy: 0.8368 - auc: 0.9230 - loss: 0.3643 - precision: 0.8282 - recall: 0.8301