import os
import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import plotly.express as px

# ✅ Charger les variables d'environnement
load_dotenv(dotenv_path='../.env')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')


# ✅ Configuration de MLflow
MLFLOW_TRACKING_URI = os.getenv("APP_URI_MLFLOW")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("InceptionV3_Metadata_sample")

# ✅ Charger le dataset
csv_path = os.getenv('CSV_PATH')
data_path = csv_path + "/merge_metadata_clean.csv"
df = pd.read_csv(data_path)

# ✅ Convertir `sex` en valeurs numériques
df["sex"] = df["sex"].str.lower().map({"female": 1, "male": 0})

# ✅ Convertir `target` en valeurs numériques
df["target"] = df["target"].str.lower().map({"benign": 0, "malignant": 1})

# ✅ Normaliser `age_approx`
df["age_approx"] = df["age_approx"] / df["age_approx"].max()
df["target"] = df["target"].astype(int)

# ✅ Prendre un échantillon de 20% du dataset
df_sampled = df.sample(frac=0.2, random_state=42).reset_index(drop=True)

# ✅ Split en train (70%) et validation (30%)
train_df, val_df = train_test_split(df_sampled, test_size=0.3, random_state=42, stratify=df_sampled["target"])

# ✅ Chemin des images
IMAGE_DIR = os.getenv('IMAGE_DIR')

# 🔹 Custom Data Generator
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

        # Charger et normaliser les images
        images = np.stack([
            tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(
                    os.path.join(self.image_dir, img_id), target_size=self.target_size
                )
            ) / 255.0 for img_id in batch_data["image_id"]
        ]).astype(np.float32)

        # Charger les métadonnées
        metadata = np.stack(batch_data[["sex", "age_approx"]].values).astype(np.float32)

        # Charger les labels
        labels = np.array(batch_data["target"].values, dtype=np.float32)

        return ({"image_input": tf.convert_to_tensor(images), 
                 "metadata_input": tf.convert_to_tensor(metadata)}, 
                tf.convert_to_tensor(labels))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_labels(self):
        return self.dataframe["target"].values

# ✅ Création des data generators
train_generator = MultiInputDataGenerator(train_df, IMAGE_DIR, shuffle=True)
val_generator = MultiInputDataGenerator(val_df, IMAGE_DIR, shuffle=False)

# ✅ Activer l’autologging COMPLET
mlflow.tensorflow.autolog()

# 🔹 Définition du modèle InceptionV3
image_input = Input(shape=(299, 299, 3), name="image_input")
base_model = InceptionV3(weights="imagenet", include_top=False, input_tensor=image_input)
base_model.trainable = True
for layer in base_model.layers[:-30]:  
    layer.trainable = False

# 🔹 Extraction de features depuis InceptionV3
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)  # Augmenter la taille du vecteur de sortie
x = Dense(128, activation="relu")(x)

# 🔹 Traitement des métadonnées (âge et sexe)
metadata_input = Input(shape=(2,), name="metadata_input")  
metadata_x = Dense(128, activation="relu")(metadata_input)
metadata_x = Dense(128, activation="relu")(metadata_x)  # Augmenter la complexité
metadata_x = Dense(128, activation="relu")(metadata_x)  # Plus de transformations

# ✅ Appliquer un poids plus élevé aux métadonnées
metadata_x = tf.keras.layers.Lambda(lambda x: x * 2.0)(metadata_x) 

# 🔹 Fusionner les features image + metadata
merged = Concatenate()([x, metadata_x])
merged = Dense(512, activation="relu")(merged)  # Augmenter la fusion
merged = Dense(256, activation="relu")(merged)

# 🔹 Couche de sortie
output = Dense(1, activation="sigmoid")(merged)

# 🔹 Compilation du modèle
model = Model(inputs=[image_input, metadata_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.00001), 
              loss="binary_crossentropy", 
              metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

# ✅ Entraînement du modèle
with mlflow.start_run():
    mlflow.log_param("train_size", len(train_df))
    mlflow.log_param("val_size", len(val_df))
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 3)

    history = model.fit(train_generator, validation_data=val_generator, epochs=3)

    val_loss, val_acc, val_precision, val_recall, val_auc = model.evaluate(val_generator)

    # ✅ Log des métriques manuelles
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("val_accuracy", val_acc)
    mlflow.log_metric("val_precision", val_precision)
    mlflow.log_metric("val_recall", val_recall)
    mlflow.log_metric("val_auc", val_auc)

    # ✅ Correction : S'assurer que `steps` est un entier
    y_true = val_generator.get_labels()
    num_samples = len(y_true)
    steps = int(np.ceil(num_samples / val_generator.batch_size))  # Convertir en `int`

    # ✅ Faire la prédiction
    y_pred_prob = model.predict(val_generator, steps=steps).ravel()

    # ✅ Ajuster `y_true` pour correspondre exactement à `y_pred_prob`
    y_true = y_true[:len(y_pred_prob)]

    # ✅ Convertir les probabilités en classes (0 ou 1)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # ✅ Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=["Bénin", "Malin"], columns=["Bénin", "Malin"])

    # ✅ Sauvegarder la matrice de confusion
    plt.figure(figsize=(6, 5))
    fig = px.imshow(df_cm, text_auto=True, color_continuous_scale="Blues")
    fig.write_image("benchmark_models/artefacts/confusion_matrix.png")

    # ✅ Log de la matrice de confusion dans MLflow
    mlflow.log_artifact("benchmark_models/artefacts/confusion_matrix.png")

    # ✅ Sauvegarde du modèle en format `.keras`
    os.makedirs("models", exist_ok=True)
    model.save("benchmark_models/artefacts/inceptionv3_skin_cancer_finetuned.keras")
    mlflow.log_artifact("benchmark_models/artefacts/inceptionv3_skin_cancer_finetuned.keras")

    print("Fine-tuning completed and logged to MLflow!")