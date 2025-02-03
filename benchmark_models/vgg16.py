import os
import pandas as pd
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
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
os.environ["APP_URI"] = os.getenv('APP_URI_MLFLOW')  # Remplace par ton URI MLflow
EXPERIMENT_NAME = "feat_VGG16_sample"

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

# Générateur pour l'ensemble d'entraînement
train_generator = img_generator.flow_from_dataframe(
    dataframe=df_sample,
    directory=sampled_image_dir,
    x_col="image_id",
    y_col="target",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=True,
    subset="training",
    seed=42
)

# Générateur pour l'ensemble de validation
val_generator = img_generator.flow_from_dataframe(
    dataframe=df_sample,
    directory=sampled_image_dir,
    x_col="image_id",
    y_col="target",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False,
    subset="validation",
    seed=42
)

# Enregistrement automatique avec MLflow
mlflow.tensorflow.autolog()

# Dynamically calculate steps per epoch
steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)

base_model  = VGG16(
                        weights='imagenet',
                        include_top=False,
                        input_shape=(224, 224, 3)
                        )

base_model.trainable = False  # Freeze the base model

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

LR = 0.001
EPOCHS = 3

model.compile(optimizer=Adam(learning_rate=LR), 
              loss='binary_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.AUC(name="auc")]
              )

#model.summary()

# Entraînement du modèle avec suivi MLflow
with mlflow.start_run() as run:
    # Enregistrement des paramètres
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("model", "VGG16")
    mlflow.log_param("dropout_rate", 0.5)
    mlflow.log_param("train_val_split", 0.7)

    # Entraînement du modèle
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    # Évaluation sur l'ensemble de validation
    y_true = val_generator.classes
    y_pred_prob = model.predict(val_generator).ravel()
    y_pred = (y_pred_prob > 0.5).astype("int32")

    # Calcul des métriques de classification
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    accuracy = history.history['val_accuracy'][-1]

    mlflow.log_metric("Validation AUC", roc_auc)
    mlflow.log_metric("Validation Accuracy", accuracy)

    # 🔹 Ajout des métriques demandées
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    mlflow.log_metric("Validation Precision", precision)
    mlflow.log_metric("Validation Recall", recall)
    mlflow.log_metric("Validation F1-score", f1)

    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1-score: {f1:.4f}")

    # Sauvegarde des courbes d'apprentissage
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Courbe de Perte")
    plt.xlabel("Époques")
    plt.ylabel("Loss")
    plt.savefig("benchmark_models/artefacts/VGG16_loss_plot.png")
    mlflow.log_artifact("benchmark_models/artefacts/VGG16_loss_plot.png")

    # Tracé et enregistrement de la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=['Bénin', 'Malin'], columns=['Bénin', 'Malin'])
    fig = px.imshow(df_cm, text_auto=True, color_continuous_scale='Blues')
    fig.write_image("benchmark_models/artefacts/VGG16_confusion_matrix.png", format="png", engine='kaleido')
    mlflow.log_artifact("benchmark_models/artefacts/VGG16_confusion_matrix.png")

    # Enregistrement du modèle
    model.save("benchmark_models/artefacts/VGG16_model.keras")
    mlflow.log_artifact("benchmark_models/artefacts/VGG16_model.keras")

    # Enregistrement du seuil optimal basé sur la courbe ROC
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    mlflow.log_param("Optimal Threshold", optimal_threshold)

    print(f"Seuil optimal basé sur le point de coude : {optimal_threshold:.4f}")

print("Enregistrement des métriques et du modèle terminé.")
