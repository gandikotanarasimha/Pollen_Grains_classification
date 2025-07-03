from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image  # for verifying uploaded images

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = load_model('model.h5')

# ✅ Actual class labels for pollen grain classification
CLASS_NAMES = [
    'arecaceae', 'asteraceae', 'chenopodiaceae', 'cupressaceae', 'cyperaceae',
    'ericaceae', 'fabaceae', 'fagaceae', 'oleaceae', 'pinaceae',
    'poaceae', 'rosaceae', 'salicaceae', 'betulaceae', 'brassicaceae',
    'caryophyllaceae', 'euphorbiaceae', 'ranunculaceae', 'polygonaceae', 'malvaceae'
]

# 🏠 Home Page
@app.route('/')
def index():
    return render_template('index.html')

# 🔍 Prediction Page
@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html', result=None, image=None)

# 📩 Contact Page
@app.route('/contact')
def contact():
    return render_template('contact.html')


# 🚪 Logout Page
@app.route('/logout.html')
def logout():
    return render_template('logout.html')

# 🤖 Result Endpoint – handles image upload & prediction
@app.route('/result', methods=['POST'])
def result():
    file = request.files.get('file')
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"[INFO] File saved: {filepath}")

        try:
            # Verify it's a valid image
            img_verify = Image.open(filepath)
            img_verify.verify()

            # Reload and preprocess for prediction
            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = model.predict(img_array)
            class_index = np.argmax(prediction[0])
            class_name = CLASS_NAMES[class_index]

            print(f"[INFO] Predicted class: {class_name}")

            return render_template('prediction.html', result=class_name, image=filename)
        except Exception as e:
            print("⚠️ Error processing image:", e)
            return "Invalid image file. Please upload a valid .jpg or .png image."

    return redirect(url_for('index'))

# 🟢 Run the app
if __name__ == '__main__':
    app.run(debug=True)
