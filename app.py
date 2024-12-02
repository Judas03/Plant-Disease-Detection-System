from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = load_model('model_path.keras')

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Supported file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Mapping of predictions to suggestion messages
disease_suggestions = {
    'Black Spot': 'Prune and discard infected leaves immediately to prevent the spread of the fungus. Avoid overhead watering, and apply a fungicide specifically for black spot as per instructions. Ensure proper spacing between plants for better air circulation.',
    'Downy Mildew': 'Remove affected foliage and apply a fungicide labeled for downy mildew. Improve air circulation around the plant by spacing or pruning, and avoid watering in the late evening to minimize humidity.',
    'Fresh Leaf': 'No signs of disease detected. Continue providing optimal care by maintaining proper watering, adequate sunlight, and regular inspections to ensure the plant remains healthy.'
}

# Predict function
def predict_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize based on your model's input size
    img_array = np.array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Model prediction
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    
    # You should have a list of class names in the same order as the model output
    class_names = ['Black Spot', 'Downy Mildew', 'Fresh Leaf']  # Example class names
    predicted_class = class_names[predicted_class_idx]
    confidence = np.max(prediction)
    
    # Get suggestion based on prediction
    suggestion = disease_suggestions.get(predicted_class, 'No suggestion available for this class.')
    
    return predicted_class, confidence, suggestion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make prediction
        predicted_class, confidence, suggestion = predict_image(filepath)

        # Return result with prediction and suggestion as JSON
        return jsonify({
            'prediction': f"Predicted Disease: {predicted_class} with confidence {confidence:.2f}",
            'suggestion': suggestion
        })
    else:
        return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)
