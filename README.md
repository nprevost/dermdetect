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

Dans le dossier streamlit, lancer les commandes suivantes pour lancer le docker
```
docker build . -t dermdetect_streamlit
docker run -it -v "$(pwd):/home/app" -e PORT=80 -p 4000:80 dermdetect_streamlit
```

## Deep Learning

Dans le fichier .env, rajouter les liens pour les dossiers des images et csv (Exemple):
```
CSV_PATH=csv
IMAGE_DIR=images
APP_URI_MLFLOW=https://nprevost-dermdetect-mlflow.hf.space
```

Pour les récuperer dans le script python
```
load_dotenv(dotenv_path='.env')
            
csv_path = os.getenv('CSV_PATH')
```
