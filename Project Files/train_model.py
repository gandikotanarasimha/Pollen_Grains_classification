# train_model.py
import os, zipfile, shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# Unzip and organize dataset
with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("extracted")

# Search for train and test folders inside the extracted structure
def find_data_folders(base_path="extracted"):
    for root, dirs, _ in os.walk(base_path):
        if 'train' in dirs and 'test' in dirs:
            return os.path.join(root, 'train'), os.path.join(root, 'test')
    return "", ""

train_dir, test_dir = find_data_folders()

# Validate folder discovery
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError("Train or test directories not found in extracted dataset.")

# Clean and move train/test folders to 'data/' directory
os.makedirs("data", exist_ok=True)
for folder in ['data/train', 'data/test']:
    if os.path.exists(folder): shutil.rmtree(folder)

shutil.move(train_dir, "data/train")
shutil.move(test_dir, "data/test")

# Image Preprocessing
IMG_SIZE = 150
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "data/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    "data/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Build the CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Train the Model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    callbacks=[early_stop]
)

model.save("model.h5")
print("[INFO] Model saved to model.h5")

# Evaluation
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

labels = list(test_generator.class_indices.keys())

print("Classification Report:\n", classification_report(y_true, y_pred_classes, target_names=labels))
cm = confusion_matrix(y_true, y_pred_classes)

  
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
