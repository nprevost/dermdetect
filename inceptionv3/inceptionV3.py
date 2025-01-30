import mlflow
import mlflow.tensorflow
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import plotly.express as px

# Charger les variables d'environnement
load_dotenv(dotenv_path='/Users/maurice/Documents/certification/dermdetect/baseline_model/.env')

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Configuration de MLflow
os.environ["APP_URI"] = "https://nprevost-dermdetect-mlflow.hf.space"
EXPERIMENT_NAME = "feat_inceptionV3"

mlflow.set_tracking_uri(os.environ["APP_URI"])
mlflow.set_experiment(EXPERIMENT_NAME)

# Définition des chemins
sample_csv_path = "/Users/maurice/Documents/certification/dermdetect/csv/sample_dataset.csv"
sampled_image_dir = "/Users/maurice/Documents/data_nogit/Dermdetect/SAMPLE_20"

# Chargement des données
df_sample = pd.read_csv(sample_csv_path)

# Génération d'images avec une simplification de l'augmentation
img_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.inception_v3.preprocess_input,
    validation_split=0.3
)

# Générateur pour l'ensemble d'entraînement
train_generator = img_generator.flow_from_dataframe(
    dataframe=df_sample,
    directory=sampled_image_dir,
    x_col="image_id",
    y_col="target",
    target_size=(299, 299),
    batch_size=32,
    class_mode="binary",
    shuffle=True,
    subset="training",
    seed=4
)

# Générateur pour l'ensemble de validation
val_generator = img_generator.flow_from_dataframe(
    dataframe=df_sample,
    directory=sampled_image_dir,
    x_col="image_id",
    y_col="target",
    target_size=(299, 299),
    batch_size=32,
    class_mode="binary",
    shuffle=True,
    subset="validation",
    seed=42
)

# Enregistrement automatique avec MLflow
mlflow.tensorflow.autolog()

# Chargement du modèle pré-entraîné InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Dégeler les 30 dernières couches pour le fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Ajouter des couches personnalisées pour la classification binaire
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x) #softmax avec 2 classes activation  = 'softmax'

# Définition du modèle final
model = Model(inputs=base_model.input, outputs=output)

# Compilation du modèle avec un learning rate réduit
model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name="auc")]
)

# Définition du callback EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entraînement du modèle avec suivi MLflow
with mlflow.start_run() as run:

    # Entraînement du modèle
    history = model.fit(
        train_generator,
        epochs=25,
        validation_data=val_generator,
        verbose=1,
        callbacks=[early_stopping]
    )

    # Évaluation sur l'ensemble de validation
    y_true = val_generator.classes
    y_pred_prob = model.predict(val_generator).ravel()
    y_pred = (y_pred_prob > 0.409).astype("int32")

    # Calcul des métriques de classification
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    accuracy = history.history['val_accuracy'][-1]

    
    # Tracé et enregistrement de la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=['Bénin', 'Malin'], columns=['Bénin', 'Malin'])
    fig = px.imshow(df_cm, text_auto=True, color_continuous_scale='Blues')
    fig.write_image("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Enregistrement du modèle
    model.save("inceptionV3_model.keras")
    mlflow.log_artifact("inceptionV3_model.keras")

    # Enregistrement du seuil optimal basé sur la courbe ROC
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    mlflow.log_param("Optimal Threshold", optimal_threshold)

    print(f"Seuil optimal basé sur le point de coude : {optimal_threshold:.4f}")

print("Enregistrement des métriques et du modèle terminé.")
