from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import models, transforms
import io
import torch.nn as nn

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'mobilenetv2_final.pth'
CLASS_NAMES = ['cat', 'dog']
CONFIDENCE_THRESHOLD = 99.0  # percentage
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # 4MB

# Load the model
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 1)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Image transforms
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

# Check allowed file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Predict class
def predict_image(image_path):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    tensor = transform_image(image_bytes)
    with torch.no_grad():
        outputs = model(tensor)
        probability = torch.sigmoid(outputs).item()

    print(f"[DEBUG] Raw Probability: {probability:.4f}")

    if probability >= 0.98:
        predicted_class = 'dog'
        confidence = round(probability * 100, 2)
    elif probability <= 0.02:
        predicted_class = 'cat'
        confidence = round((1 - probability) * 100, 2)
    else:
        predicted_class = 'unknown'
        confidence = round(probability * 100, 2)

    return predicted_class, confidence

# Main route
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            predicted_class, confidence = predict_image(filepath)

            # Optional: delete the uploaded file after prediction
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"[WARNING] Could not delete file: {e}")

            return jsonify({
                'filename': filename,
                'prediction': predicted_class,
                'confidence': confidence,
                'class_names': CLASS_NAMES + ['unknown']
            })

        return jsonify({'error': 'File type not allowed'}), 400

    return render_template('index.html')

@app.route('/class_names', methods=['GET'])
def get_class_names():
    return jsonify({'class_names': CLASS_NAMES + ['unknown']})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
