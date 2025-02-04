import os
import pandas as pd
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

# Charger les variables d'environnement
load_dotenv(dotenv_path='../.env')

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Configuration de MLflow
os.environ["APP_URI"] = os.getenv('APP_URI_MLFLOW')
EXPERIMENT_NAME = "feat_resnet50_sample"
mlflow.set_tracking_uri(os.environ["APP_URI"])
mlflow.set_experiment(EXPERIMENT_NAME)

csv_path = os.getenv('CSV_PATH')
image_dir = os.getenv('IMAGE_DIR')

# Définition des chemins
sample_csv_path = csv_path + "/sample_dataset.csv"
sampled_image_dir = image_dir + "/SAMPLE_20"

# Chargement des données
df_sample = pd.read_csv(sample_csv_path)

# Génération d'images avec preprocessing
img_generator = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.3
)

# Générateurs d'images
train_generator = img_generator.flow_from_dataframe(
    dataframe=df_sample,
    directory=sampled_image_dir,
    x_col="image_id",
    y_col="target",
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    shuffle=True,
    subset="training",
    seed=42
)

val_generator = img_generator.flow_from_dataframe(
    dataframe=df_sample,
    directory=sampled_image_dir,
    x_col="image_id",
    y_col="target",
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    shuffle=False,
    subset="validation",
    seed=42
)

# Définition du modèle avec ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze du modèle de base

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

# Compilation du modèle
LR = 0.001
EPOCHS = 3
model.compile(optimizer=Adam(learning_rate=LR), 
              loss='binary_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.AUC(name="auc")])

# Enregistrement automatique avec MLflow
mlflow.tensorflow.autolog()

# Entraînement du modèle avec suivi MLflow
with mlflow.start_run() as run:
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", EPOCHS)

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        verbose=1
    )

    # Évaluation
    y_true = val_generator.classes
    y_pred_prob = model.predict(val_generator).ravel()
    y_pred = (y_pred_prob > 0.5).astype("int32")

    # Calcul des métriques
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    accuracy = history.history['val_accuracy'][-1]

    # Enregistrement des métriques dans MLflow
    mlflow.log_metric("Validation AUC", roc_auc)
    mlflow.log_metric("Validation Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1-score", f1)

    # Sauvegarde des courbes d'apprentissage
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Courbe de Perte")
    plt.xlabel("Époques")
    plt.ylabel("Loss")
    plt.savefig("benchmark_models/artefacts/resnet50_loss_plot.png")
    mlflow.log_artifact("benchmark_models/artefacts/resnet50_loss_plot.png")

    # Tracé de la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=['Bénin', 'Malin'], columns=['Bénin', 'Malin'])
    fig = px.imshow(df_cm, text_auto=True, color_continuous_scale='Blues')
    fig.write_image("benchmark_models/artefacts/resnet50_confusion_matrix.png", format="png", engine='kaleido')
    mlflow.log_artifact("benchmark_models/artefacts/resnet50_confusion_matrix.png")

    # Enregistrement du modèle
    model.save("benchmark_models/artefacts/resnet50_model.keras")
    mlflow.log_artifact("benchmark_models/artefacts/resnet50_model.keras")

    # Seuil optimal basé sur ROC
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    mlflow.log_param("Optimal Threshold", optimal_threshold)
    print(f"Seuil optimal basé sur le point de coude : {optimal_threshold:.4f}")

    # Affichage des métriques principales
    print(f"Validation AUC: {roc_auc:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

print("Enregistrement des métriques et du modèle terminé.")
