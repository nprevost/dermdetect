import pandas as pd
import json
import os
from dotenv import load_dotenv
from pathlib import Path
import shutil
import requests
import time

load_dotenv(dotenv_path='.env')

csv_path = os.getenv('CSV_PATH')
image_dir = os.getenv('IMAGE_DIR')

# Fonction pour deplacer les images
def copy_images(source):
    if os.path.exists(source+"/DA"):
        noms_images = []
        for root, dirs, files in os.walk(source+"/DA"):
            for file in files:
                if file.lower().endswith('.jpg'):
                    source_file = os.path.join(root, file)
                    destination_file = os.path.join(source, file)
                    noms_images.append(file)

                    # Récupérer les noms des fichiers image
                    shutil.move(source_file, destination_file)
                    
                    print(f"Copied: {source_file} to {destination_file}")

        # Création du dataset avec image_id
        df = pd.DataFrame(noms_images, columns=['image_id'])
        fichier_csv = csv_path + "/dataset.csv"
        df.to_csv(fichier_csv, index=False)

        shutil.rmtree(source+"/DA")
        print("All images copied successfully!")

def clean_dataset(metadata):
    metadata['image_id'] = metadata['image_id'].str.replace(r'_\d+\.JPG$', '', regex=True)

    # Nombre initial de lignes
    initial_count = len(metadata)

    # Supprimer les doublons basés sur les colonnes image_id et target
    metadata = metadata.drop_duplicates(subset=['image_id']).reset_index(drop=True)

    # Nombre de lignes après suppression des doublons
    final_count = len(metadata)

    # Calculer le nombre de lignes supprimées
    lines_removed = initial_count - final_count

    # Afficher le résultat
    print(f"Nombre de lignes supprimées : {lines_removed}")
    return metadata

# Fonction pour récupérer les métadonnées avec retry
def fetch_metadata(image_id, base_url, headers, retries=3, delay=2):
    url = f"{base_url}{image_id}/"
    for attempt in range(retries):
        try:
            print(f"Call : {url}")
            response = requests.get(url, headers=headers, timeout=10)  # Timeout ajouté
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Erreur pour {image_id}: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Tentative {attempt + 1} échouée pour {image_id}: {e}")
            time.sleep(delay)  # Attente avant la prochaine tentative
    print(f"Échec définitif pour {image_id} après {retries} tentatives.")
    return None

def create_df_metadata(json_data):
    # Désimbriquer et convertir en DataFrame
    df = pd.json_normalize(json_data)

    # Sélectionner et renommer les colonnes souhaitées
    df_final = df[[
        "isic_id",
        "copyright_license",
        "files.full.url",
        "files.full.size",
        "files.thumbnail_256.url",
        "files.thumbnail_256.size",
        "metadata.acquisition.pixels_x",
        "metadata.acquisition.pixels_y",
        "metadata.acquisition.image_type",
        "metadata.clinical.concomitant_biopsy",
        "metadata.clinical.sex",
        "metadata.clinical.anatom_site_general",
        "metadata.clinical.benign_malignant",
        "metadata.clinical.diagnosis_1",
        "metadata.clinical.diagnosis_confirm_type",
        "metadata.clinical.age_approx",
        "metadata.clinical.lesion_id",
        "metadata.clinical.patient_id"
    ]]

    # Renommer les colonnes pour plus de clarté
    df_final.columns = [
        "isic_id",
        "copyright_licence",
        "full_url",
        "full_size",
        "256_url",
        "256_size",
        "pixels_x",
        "pixels_y",
        "image_type",
        "concomitant_biopsy",
        "sex",
        "anatom_site_general",
        "target",
        "diagnosis_1",
        "diagnosis_confirm_type",
        "age_approx",
        "lesion_id",
        "patient_id"
    ]

    # Définir le chemin de sortie pour le fichier CSV
    csv_output_path = csv_path + "/metadata.csv"

    # Sauvegarder en CSV
    df_final.to_csv(csv_output_path, index=False)

    # Afficher un message de confirmation avec le chemin du fichier CSV
    print(csv_output_path)

###########################################
if os.path.exists(image_dir+"/DA"):
    copy_images(image_dir)

if os.path.isfile(csv_path+"/dataset.csv"):
    metadata = pd.read_csv(csv_path + '/dataset.csv')
    
    metadata = clean_dataset(metadata)


    # Liste des image_id dans le DataFrame `metadata`
    image_ids = metadata["image_id"].tolist()

    # Charger les métadonnées existantes si elles sont déjà enregistrées
    output_file = csv_path + "/metadata_partial.json"

    try:
        with open(output_file, "r") as json_file:
            all_metadata = json.load(json_file)
            # Correction ici : utiliser "isic_id" au lieu de "id"
            fetched_ids = {entry["isic_id"] for entry in all_metadata}
    except (FileNotFoundError, json.JSONDecodeError):
        all_metadata = []
        fetched_ids = set()

    # URL de base de l'API
    base_url = "https://api.isic-archive.com/api/v2/images/"

    # En-têtes de requête
    headers = {
        "accept": "application/json",
        "X-CSRFToken": "rFjRY2QusJwfWoXaHS7zsTd6dpMRFjG3syKTIAB1xH9ivj5GKHNovnjlxY5qoDWO"
    }

    # Collecter les métadonnées pour tous les image_id restants
    for idx, image_id in enumerate(image_ids):
        if image_id in fetched_ids:
            continue  # Sauter les images déjà traitées
        print(f"Fetching metadata for image_id: {image_id} ({idx + 1}/{len(image_ids)})")
        metadata_entry = fetch_metadata(image_id, base_url, headers)
        if metadata_entry:  # Ajouter seulement si la requête est réussie
            all_metadata.append(metadata_entry)
            fetched_ids.add(image_id)

        # Sauvegarder toutes les 100 requêtes
        if len(all_metadata) % 100 == 0:
            with open(output_file, "w") as json_file:
                json.dump(all_metadata, json_file, indent=4)
            print(f"Progress saved after {len(all_metadata)} entries.")

    with open(output_file, "w") as json_file:
        json.dump(all_metadata, json_file, indent=4)
    print(f"Progress saved after {len(all_metadata)} entries.")

    create_df_metadata(all_metadata)

    os.remove(csv_path + '/dataset.csv')