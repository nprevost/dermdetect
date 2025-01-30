import os
import shutil

# Define paths
ben_dir = r"D:\Desktop\dermproject\data\BEN"
mal_dir = r"D:\Desktop\dermproject\data\MAL"
all_images_dir = r"D:\Desktop\dermproject\data\ALL_IMAGES"

# Create ALL_IMAGES folder if it doesn't exist
os.makedirs(all_images_dir, exist_ok=True)

# Move images from BEN to ALL_IMAGES
for file in os.listdir(ben_dir):
    shutil.move(os.path.join(ben_dir, file), os.path.join(all_images_dir, file))

# Move images from MAL to ALL_IMAGES
for file in os.listdir(mal_dir):
    shutil.move(os.path.join(mal_dir, file), os.path.join(all_images_dir, file))

print("All images have been moved to ALL_IMAGES!")
