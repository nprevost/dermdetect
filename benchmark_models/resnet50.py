import os
import pandas as pd
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, roc_curve, auc
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

# Générateur pour l'ensemble d'entraînement
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

# Générateur pour l'ensemble de validation
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

base_model  = ResNet50(
                        weights='imagenet',
                        include_top=False,
                        input_shape=(128, 128, 3)
                        )

base_model.trainable = False  # Freeze the base model

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
#x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

LR = 0.001
EPOCHS = 3

model.compile(optimizer=Adam(learning_rate=LR), 
              loss='binary_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.AUC(name="auc")]
              )

#plot_model(model, show_shapes=True)

# Enregistrement automatique avec MLflow
mlflow.tensorflow.autolog()

# Entraînement du modèle avec suivi MLflow
with mlflow.start_run() as run:
    # Enregistrement des paramètres
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", EPOCHS)

    # Entraînement du modèle
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        verbose=1
    )

    # Évaluation sur l'ensemble de validation
    y_true = val_generator.classes
    y_pred_prob = model.predict(val_generator).ravel()
    y_pred = (y_pred_prob > 0.5).astype("int32")

    # Calcul des métriques de classification
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    accuracy = history.history['val_accuracy'][-1]

    #mlflow.log_metric("Validation AUC", roc_auc)
    #mlflow.log_metric("Validation Accuracy", accuracy)

    # Sauvegarde des courbes d'apprentissage
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Courbe de Perte")
    plt.xlabel("Époques")
    plt.ylabel("Loss")
    plt.savefig("model/resnet50_loss_plot.png")
    mlflow.log_artifact("model/resnet50_loss_plot.png")

    # Tracé et enregistrement de la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=['Bénin', 'Malin'], columns=['Bénin', 'Malin'])
    fig = px.imshow(df_cm, text_auto=True, color_continuous_scale='Blues')
    fig.write_image("model/resnet50_confusion_matrix.png", format="png", engine='kaleido')
    mlflow.log_artifact("model/resnet50_confusion_matrix.png")

    # Enregistrement du modèle
    model.save("model/resnet50_model.keras")
    mlflow.log_artifact("model/resnet50_model.keras")


    # Enregistrement du seuil optimal basé sur la courbe ROC
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    mlflow.log_param("Optimal Threshold", optimal_threshold)

    print(f"Seuil optimal basé sur le point de coude : {optimal_threshold:.4f}")

print("Enregistrement des métriques et du modèle terminé.")