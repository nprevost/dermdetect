# Dermdetect

Dermdetect is a project to identify skin cancers from images. Using the Inception_v3 model and developed with Streamlit, this application allows you to upload images of skin lesions and get fast and accurate predictions.

---

## Stack used
- **MLFlow**: Used for lifecycle management of machine learning models.
- **Streamlit**: Used to build the web UI.
- **scikit-learn**: Used to implement the inception_v3.
- **pandas**: For data manipulation.
- **plotly**: For creating interactive visualizations.
- **S3**: To save artifacts 

---

## Startup

### Installation

```bash
   git clone https://github.com/nprevost/dermdetect.git
   cd dermdetect
```

---

### Make an environment

```
python3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Copier le fichier .env.tmp, renommer en .env et remplir les infos.

---

### Download images
[Images](https://www.kaggle.com/datasets/ernestbeckham/skin-cancer-shim-preprocessed/data)

### Preparation
- First you need to create, put the zip and unzip it in an images folder of the project.
- Then create a csv with the name of the images and the target. Group all the images in the same folder and retrieve the information with the ISIC API. To do this, launch **preprocessing.py** which is in the scripts folder.

---

### MLFlow

[URL du hugging face](https://huggingface.co/spaces/nprevost/dermdetect-mlflow)

[URL](https://nprevost-dermdetect-mlflow.hf.space/)

You need to create a database with neon db

Also create an S3 bucket

---

### Streamlit
[URL du hugging face](https://huggingface.co/spaces/nprevost/dermdetect-streamlit/tree/main)

[URL](https://nprevost-dermdetect-streamlit.hf.space/)

Dans le dossier streamlit, lancer les commandes suivantes pour lancer le docker
```
docker build . -t dermdetect_streamlit
docker run -it -v "$(pwd):/home/app" -e PORT=80 -p 4000:80 dermdetect_streamlit
```

---

## Deep Learning

Dans le fichier .env, rajouter le lien pour MLFlow (Exemple):
```
APP_URI_MLFLOW=https://nprevost-dermdetect-mlflow.hf.space
```

Pour les récuperer dans le script python
```
load_dotenv(dotenv_path='.env')
            
csv_path = os.getenv('CSV_PATH')
```

## File Structure

```
dermdetect/
├── .venv/                       # Environment python
├── benchmark_models/            # Model used to analyze performance
│   ├── artefacts                # Models and matrix saved
│   ├── baseline_cnn.py          # Model base CNN
│   ├── resnet50.py              # Model resnet50
│   ├── vgg16.py                 # Model VGG16
│   └── inceptionv3.py           # Model inception_v3
├── csv/                         # Datasets
├── images/                      # Images for training
├── mlflow/                      # Files for mlflow
│   ├── Dockerfile               # Docker configuration file
│   ├── README.md                # Config MLFlow
│   └── requirements.txt         # Dependencies for MLFlow
├── model/                       # Best model
├── scripts/                     # Scripts python
│   ├── preprocessing.py         # Scripts to clean images and dataset
│   ├── sample.py                # Scripts to create sample images
│   └── script_test_mlflow.py    # Small script to test the connection to mlflow
├── streamlit/                   # Files for streamlit
│   ├── .streamlit/
│   │   └── config.toml          # Config streamlit
│   ├── pages/                   # Different pages of the streamlit
│   |   ├── dataset.py           # Page to analyze the dataset
│   │   ├── intro.py             # Streamlit Home Page
│   |   └── model.py             # Page to make the prediction
│   ├── Dockerfile               # Docker configuration file
|   |── requirements.txt         # Dependencies for streamlit
│   └── streamlit_app.py         # Menu and page management
├── .env                         # Environment variables local 
├── .env.tmp                     # Environment variables Example
├── .gitignore                   # Git ignore file
├── README.md                    # Project documentation
└── requirements.txt             # Dependencies for the project
```