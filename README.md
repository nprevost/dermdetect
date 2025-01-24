# dermdetect

## MLFlow

url du hugging face = https://huggingface.co/spaces/nprevost/dermdetect-mlflow

url = https://nprevost-dermdetect-mlflow.hf.space/

Créer un fichier .env au niveau du script_test_mlflow.py

```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
```

## Streamlit

Url du hugging face = https://huggingface.co/spaces/nprevost/dermdetect-streamlit/tree/main

url = https://nprevost-dermdetect-streamlit.hf.space/

## Deep Learning

Dans le fichier .env, rajouter les liens pour les dossiers des images et csv (Exemple):
```
CSV_PATH=csv/dataset.csv
IMAGE_DIR=images
SAMPLED_IMAGE_DIR=SAMPLE_20
SAMPLE_CSV_PATH=csv/sample_dataset.csv
```

Pour les récuperer dans le script python
```
load_dotenv(dotenv_path='.env')
            
csv_path = os.getenv('CSV_PATH')
```
