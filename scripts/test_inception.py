import os
import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.keras
from mlflow.models.signature import ModelSignature, Schema
from mlflow.types.schema import TensorSpec
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ---------------------
# Environment & MLflow Setup
# ---------------------
load_dotenv(dotenv_path="../.env")  # Adjust as needed

MLFLOW_TRACKING_URI = os.getenv("APP_URI_MLFLOW")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("SkinCancer_Exp1_InceptionV3_Metadata")

# ---------------------
# Data Loading & Preprocessing
# ---------------------
data_path = os.getenv("MERGED_CLEANED_DATASET")
df = pd.read_csv(data_path)

df["sex"] = df["sex"].str.lower().map({"female": 1, "male": 0})
df["age_approx"] = df["age_approx"] / df["age_approx"].max()
df_sampled = df.sample(frac=0.2, random_state=42).reset_index(drop=True)
train_df, val_df = train_test_split(df_sampled, test_size=0.3, random_state=42, stratify=df_sampled["target"])
train_df["target"] = train_df["target"].astype(str)
val_df["target"] = val_df["target"].astype(str)
IMAGE_DIR = os.getenv("IMAGE_DIR")

# ---------------------
# Custom Data Generator
# ---------------------
class MultiInputDataGenerator(Sequence):
    def __init__(self, dataframe, image_dir, batch_size=32, target_size=(299, 299), shuffle=True, **kwargs):
        super().__init__(**kwargs)
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

        images = np.stack([
            tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(
                    os.path.join(self.image_dir, img_id),
                    target_size=self.target_size
                )
            ) / 255.0 for img_id in batch_data["image_id"]
        ]).astype(np.float32)

        metadata = np.stack(batch_data[["sex", "age_approx"]].values).astype(np.float32)
        labels = np.array(batch_data["target"].astype(float).values, dtype=np.float32)

        return (tf.convert_to_tensor(images), tf.convert_to_tensor(metadata)), tf.convert_to_tensor(labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# ---------------------
# Create Data Generators
# ---------------------
train_generator = MultiInputDataGenerator(train_df, IMAGE_DIR, shuffle=True)
val_generator = MultiInputDataGenerator(val_df, IMAGE_DIR, shuffle=False)

# ---------------------
# Build the Multi-Input Model
# ---------------------
image_input = Input(shape=(299, 299, 3), name="image_input")
base_model = InceptionV3(weights="imagenet", include_top=False, input_tensor=image_input)
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
x = GlobalAveragePooling2D()(base_model.output)
metadata_input = Input(shape=(2,), name="metadata_input")
merged = Concatenate()([x, metadata_input])
x = Dense(256, activation="relu")(merged)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)
model = Model(inputs=[image_input, metadata_input], outputs=output)
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
)

# ---------------------
# Input Example & Model Signature for MLflow
# ---------------------
sample_images = np.random.rand(1, 299, 299, 3).astype(np.float32)
sample_metadata = np.random.rand(1, 2).astype(np.float32)

# MLflow model signature with named TensorSpecs (using -1 for the dynamic batch dimension)
image_input_spec = TensorSpec(shape=(-1, 299, 299, 3), type=np.dtype(np.float32), name="image_input")
metadata_input_spec = TensorSpec(shape=(-1, 2), type=np.dtype(np.float32), name="metadata_input")
output_spec = TensorSpec(shape=(-1,), type=np.dtype(np.float32), name="predictions")
input_schema = Schema([image_input_spec, metadata_input_spec])
output_schema = Schema([output_spec])
model_signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# ---------------------
# Train, Evaluate, and Log the Model via MLflow
# ---------------------
with mlflow.start_run():
    mlflow.log_param("train_size", len(train_df))
    mlflow.log_param("val_size", len(val_df))
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 1)
    mlflow.log_param("fine_tune_layers", 30)

    history = model.fit(train_generator, validation_data=val_generator, epochs=1)
    val_loss, val_acc, val_precision, val_recall, val_auc = model.evaluate(val_generator)
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("val_accuracy", val_acc)
    mlflow.log_metric("val_precision", val_precision)
    mlflow.log_metric("val_recall", val_recall)
    mlflow.log_metric("val_auc", val_auc)
    print("Fine-tuning completed and logged to MLflow!")

    # With the upgraded MLflow, you can now use save_input_example to disable saving input examples if desired.
    # If you want to save the input example, remove the parameter or set it to True.
    mlflow.keras.log_model(
        model,
        "keras_model",
        input_example={"image_input": sample_images, "metadata_input": sample_metadata},
        signature=model_signature
    )

    # Generate predictions on validation data and log a confusion matrix artifact
    y_true, y_pred = [], []
    for (images, metadata), labels in val_generator:
        batch_preds = model.predict([images, metadata])
        y_true.extend(labels.numpy())
        y_pred.extend(batch_preds.flatten())
    y_true = np.array(y_true)
    y_pred = (np.array(y_pred) > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_image_path = "confusion_matrix.png"
    plt.savefig(cm_image_path)
    mlflow.log_artifact(cm_image_path)
    print("Confusion matrix logged to MLflow!")

# ---------------------
# Save the Model Locally
# ---------------------
os.makedirs("../models", exist_ok=True)
model.save("../models/inceptionv3_skin_cancer_finetuned.keras")
print("I'm done!")
