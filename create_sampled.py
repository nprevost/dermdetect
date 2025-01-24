import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def create_sampled_dataset(base_dir, sampled_dir, train_dir, val_dir, sample_frac=0.2):
    if os.path.exists(sampled_dir):
        shutil.rmtree(sampled_dir)  # Clean up any previous runs
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Create lists of all images
    data = []
    for label, folder in enumerate(["BEN", "MAL"]):
        folder_path = os.path.join(base_dir, folder)
        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                data.append((os.path.join(folder_path, img_name), label))
    
    # Convert to DataFrame and sample
    data_df = pd.DataFrame(data, columns=["image_path", "label"])
    sampled_df = data_df.sample(frac=sample_frac, random_state=42)

    # Split sampled data into train and validation sets
    train_df, val_df = train_test_split(sampled_df, test_size=0.3, random_state=42, stratify=sampled_df["label"])

    # Copy sampled images to the new directories
    for subset, subset_dir in [(train_df, train_dir), (val_df, val_dir)]:
        for _, row in subset.iterrows():
            label_folder = "BEN" if row["label"] == 0 else "MAL"
            dest_folder = os.path.join(subset_dir, label_folder)
            os.makedirs(dest_folder, exist_ok=True)
            shutil.copy(row["image_path"], dest_folder)