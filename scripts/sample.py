import pandas as pd
import os
from dotenv import load_dotenv
import shutil

# Charger les variables d'environnement
load_dotenv(dotenv_path='../.env')

# Définition des chemins
csv_path = os.getenv('CSV_PATH') + "/dataset.csv"
image_dir = os.getenv('IMAGE_DIR')
sampled_image_dir = os.getenv('IMAGE_DIR') + "/SAMPLE_20"
sample_csv_path = os.getenv('CSV_PATH') + "/sample_dataset.csv"

# Charger le fichier CSV
df = pd.read_csv(csv_path)

# Normalisation des noms d'images (suppression des espaces et conversion en minuscules)
df['image_id'] = df['image_id'].str.strip().str.lower()

# Vérification des fichiers disponibles dans le dossier
files_in_directory = [f.lower() for f in os.listdir(image_dir)]

# Ajouter le chemin complet et filtrer les fichiers existants
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(image_dir, x))
df = df[df['image_path'].apply(lambda x: os.path.basename(x) in files_in_directory)]

print(f"Nombre total d'images disponibles après correspondance : {len(df)}")

# Échantillonnage de 20 % avec une graine fixe pour reproductibilité
SAMPLE_SIZE = 0.20  # 20% des données
SEED = 42  # Graine fixe

df_sample = df.sample(frac=SAMPLE_SIZE, random_state=SEED)

print(f"Nombre d'images dans l'échantillon de 20% : {len(df_sample)}")

# Créer le dossier pour stocker l'échantillon si nécessaire
if not os.path.exists(sampled_image_dir):
    os.makedirs(sampled_image_dir)

# Copier les images sélectionnées dans le dossier d'échantillon
for _, row in df_sample.iterrows():
    src_path = row['image_path']
    dest_path = os.path.join(sampled_image_dir, os.path.basename(row['image_id']))
    shutil.copy(src_path, dest_path)

print(f"Les images de l'échantillon ont été copiées dans : {sampled_image_dir}")

# Sauvegarder l'échantillon dans un fichier CSV pour les étapes suivantes
df_sample[["image_id", "target"]].to_csv(sample_csv_path, index=False)

print(f"CSV de l'échantillon sauvegardé dans : {sample_csv_path}")