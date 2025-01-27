import os
import pandas as pd
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve, auc
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


# Charger les variables d'environnement
load_dotenv(dotenv_path='../.env')

print(os.getenv('AWS_ACCESS_KEY_ID'))