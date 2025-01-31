import pandas as pd
import os
from dotenv import load_dotenv
import re

load_dotenv(dotenv_path='.env')

csv_path = os.getenv('CSV_PATH')

metadata = pd.read_csv(csv_path + '/metadata.csv')
dataset = pd.read_csv(csv_path + '/dataset.csv')

metadata = metadata[['isic_id', 'target', 'age_approx', 'sex']]

dataset = dataset[['image_id']]
dataset['isic_id'] = dataset['image_id'].apply(lambda x: re.sub(r'_\d+\.JPG$', '', x).upper())

merge_metadata = pd.merge(dataset, metadata, on = 'isic_id', how = 'left')

merge_metadata.to_csv(csv_path + '/merge_metadata.csv', index=False)

merge_metadata = merge_metadata.dropna()
merge_metadata.to_csv(csv_path + '/merge_metadata_clean.csv', index=False)
