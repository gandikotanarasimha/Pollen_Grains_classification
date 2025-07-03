# Pollen_Grains_classification
# Deep learning-based web application to classify pollen grain images into various plant families using CNN and Flask.

# Pollen’s Profiling – Automated Classification of Pollen Grains

**Category:** Artificial Intelligence | **Level:** Intermediate  
**Skills:** Python • Deep Learning (CNN) • Flask • Data Preprocessing • Web Deployment

## 🧭 Project Overview

Automated classification of pollen grain images using CNNs and Flask. Useful for:

- 🌱 **Environmental Monitoring**
- 🏥 **Allergy Diagnosis**

## 🚀 Features

- CNN model trained on diverse pollen datasets
- Image upload and prediction via Flask web server
- Clean and responsive UI using HTML, CSS, and JS
- Supports over 30+ pollen grain classes

📂 Tech Stack: Python, TensorFlow/Keras, Flask, HTML, CSS, JavaScript

## 🛠️ Installation & Setup

```bash
git clone https://github.com/your_username/pollen-profiling.git
cd pollen-profiling
conda create -n pollen python=3.9
conda activate pollen
pip install -r requirements.txt
python train.py --epochs 25 --batch_size 32
cd flask_app
python app.py
```

## 👣 Project Structure

```
POLLEN_GRAIN_CLASSIFIER/
│
│
├── static/
│ ├── css/
│ │ └── main.css # Styling for the frontend
│ ├── js/
│ │ └── main.js # Optional JavaScript
│ ├── images/ # Sample or background images
│ └── uploads/ # Folder to store uploaded test images
│
├── templates/
│ ├── index.html # Home page
│ ├── prediction.html # Image upload + result
│ ├── contact.html # Contact page
│ └── logout.html # Exit or contact link
│
├── uploads/
│ └── [runtime uploads go here]
│
├── model.h5 # Trained Keras model
├── train_model.py # Model training script
├── app.py # Flask backend application
├── README.md # Project documentation
└── dataset.zip # Dataset (if needed)
```
```
dataset format
dataset.zip
└── dataset/
    └── data/
        ├── train/
        │   ├── class1/
        │   │   ├── image1.jpg
        │   │   ├── image2.jpg
        │   │   └── ...
        │   ├── class2/
        │   │   ├── image1.jpg
        │   │   ├── image2.jpg
        │   │   └── ...
        │   └── ... (more classes if needed)
        └── test/
            ├── class1/
            │   ├── image1.jpg
            │   ├── image2.jpg
            │   └── ...
            ├── class2/
            │   ├── image1.jpg
            │   ├── image2.jpg
            │   └── ...
            └── ... (more classes if needed)

```
## 🧠 Results

- **Accuracy:** 95.6%
- Confusion matrix, metrics: see TECHNICAL_DOC.md

## 📚 License

MIT License — see `LICENSE`.
