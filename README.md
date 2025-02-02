# Dermdetect

Dermdetect is a project to identify skin cancers from images. Using the Inception_v3 model and developed with Streamlit, this application allows you to upload images of skin lesions and get fast and accurate predictions.

---

## Stack used
- **MLFlow**: Used for lifecycle management of machine learning models.
- **Streamlit**: Used to build the web UI.
- **Tensorflow**: Used to implement the inception_v3.
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
Copy the .env.tmp file, rename to .env and fill in the info.

---

### Download images
[Images](https://www.kaggle.com/datasets/ernestbeckham/skin-cancer-shim-preprocessed/data)

### Preparation
- First you need to create, put the zip and unzip it in an images folder of the project.
- Then create a csv with the name of the images and the target. Group all the images in the same folder and retrieve the information with the ISIC API. To do this, launch **preprocessing.py** and **clean_metadata.py** which is in the scripts folder.

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

- In the streamlit folder, create a **model** folder and add the model.keras file to it
- Add the merge_metadata.csv in the S3 and change the value of url_csv in the dataset.py file
- In the streamlit folder, run the following commands to launch the docker

```
docker build . -t dermdetect_streamlit
docker run -it -v "$(pwd):/home/app" -e PORT=80 -p 4000:80 dermdetect_streamlit
```

---

## Deep Learning

In the .env file, add the link for MLFlow (Example):
```
APP_URI_MLFLOW=https://nprevost-dermdetect-mlflow.hf.space
```

To retrieve them in the python script
```
load_dotenv(dotenv_path='.env')
            
csv_path = os.getenv('CSV_PATH')
```

## File Structure

```
dermdetect/
├── .venv/                       # Environment python
├── benchmark_models/            # Model used to analyze performance
│   ├── artefacts/               # Models and matrix saved
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
│   ├── clean_metadata.py        # Scripts to create clean metadata
│   ├── preprocessing.py         # Scripts to create dataset and move images
│   └── script_test_mlflow.py    # Small script to test the connection to mlflow
├── streamlit/                   # Files for streamlit
│   ├── .streamlit/
│   │   └── config.toml          # Config streamlit
│   ├── model/                   # Best model to predict in streamlit
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