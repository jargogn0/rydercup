import os
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
import tensorflow as tf
from imgaug import augmenters as iaa

# ----------- Step 1: Labeling -----------

# Path to your local folder containing B3 and B4 images
dir_path = "/Users/doudou.ba/Downloads/niokolokoba"

# Retrieve B3 and B4 file paths
b3_files = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if 'B3' in f])
b4_files = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if 'B4' in f])

# Function to extract patches based on NDVI values
def extract_patches(image, patch_size=50, stride=25, threshold=0.5):
    forest_patches = []
    non_forest_patches = []
    for i in range(0, image.shape[0] - patch_size, stride):
        for j in range(0, image.shape[1] - patch_size, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            mean_ndvi = np.mean(patch)
            if mean_ndvi > threshold:
                forest_patches.append(patch)
            elif mean_ndvi < 0.2:
                non_forest_patches.append(patch)
    return forest_patches, non_forest_patches

# Loop through B3 and B4 files, compute NDVI, and extract patches
all_forest_patches = []
all_non_forest_patches = []
for b3_path, b4_path in zip(b3_files, b4_files):
    with rasterio.open(b4_path) as nir_src:
        nir_data = nir_src.read(1)
    with rasterio.open(b3_path) as red_src:
        red_data = red_src.read(1)
    ndvi_data = (nir_data - red_data) / (nir_data + red_data + 1e-8)
    
    forest_patches, non_forest_patches = extract_patches(ndvi_data)
    all_forest_patches.extend(forest_patches)
    all_non_forest_patches.extend(non_forest_patches)


# ----------- Step 2: Data Augmentation -----------

# Define augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Rot90((1, 3))
])

forest_patches_augmented = seq(images=all_forest_patches)
non_forest_patches_augmented = seq(images=all_non_forest_patches)

# Combine augmented forest and non-forest patches and create labels
X = forest_patches_augmented + non_forest_patches_augmented
y = [1]*len(forest_patches_augmented) + [0]*len(non_forest_patches_augmented)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ----------- Step 3: Model Training -----------

# Model definition
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32, validation_data=(np.array(X_test), np.array(y_test)))


# ----------- Step 4: Evaluation -----------

# You can evaluate the model on the test set using:
test_loss, test_accuracy = model.evaluate(np.array(X_test), np.array(y_test))
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# You can also use other metrics like F1-score, Precision, Recall, etc., but that would require additional code.
