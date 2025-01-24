from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from create_sampled import create_sampled_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
base_dir = "../data/raw"  # Original dataset directory with BEN and MAL
sampled_dir = "../data/sample"  # Directory for sampled data
train_dir = os.path.join(sampled_dir, "train")
val_dir = os.path.join(sampled_dir, "val")

# Create the sampled dataset
create_sampled_dataset(base_dir, sampled_dir, train_dir, val_dir)

# Step 2: Set Up Image Generators
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)


base_model  = ResNet50(
                        weights='imagenet',
                        include_top=False,
                        input_shape=(224,224,3)
                        )
base_model.trainable = False

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the model with additional metrics
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])

base_model.summary()