from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
 
# Initialize Flask app
app = Flask(__name__)
 
# Load trained model
model = load_model("plant_disease_model.h5")
 
# Class labels (update with your dataset classes)
class_labels = [
    "Potato___healthy",
    "Potato___Late_blight",
    "Potato___Early_blight",
    "Tomato___healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight"
    # Add all classes from your dataset
]
 
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = image.load_img(file, target_size=(128,128))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
 
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0]))
 
    return jsonify({
        "class": class_labels[class_idx],
        "confidence": confidence
    })
 
if __name__ == "__main__":
    app.run(debug=True)