import sys
import os
import random
from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import io

# Initialize Flask app with absolute paths
app = Flask(__name__, template_folder=r"C:\Users\david\Documents\NFT_Analysis-master\nft_app\templates")

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define paths
PATHS = {
    "linear_regression": r"C:\Users\david\Documents\NFT_Analysis-master\nft_app\models\linear_regression_model.pkl",
    "knn": r"C:\Users\david\Documents\NFT_Analysis-master\nft_app\models\knn_model.pkl",
    "lstm": r"C:\Users\david\Documents\NFT_Analysis-master\nft_app\models\lstm_model.h5",
    "dataset": r"C:\Users\david\Documents\NFT_Analysis-master\nft_app\data\Processed_OpenSea_NFT_1_Sales.csv",
    "index": "index.html"
}

# Load ML models
print("\n=== Loading Machine Learning Models ===")
models = {}
try:
    models['linear_regression'] = pickle.load(open(PATHS["linear_regression"], "rb"))
    print(f"✓ Linear Regression model loaded")
except Exception as e:
    print(f"✗ Failed to load Linear Regression model: {e}")

try:
    models['knn'] = pickle.load(open(PATHS["knn"], "rb"))
    print(f"✓ KNN model loaded")
except Exception as e:
    print(f"✗ Failed to load KNN model: {e}")

try:
    models['lstm'] = load_model(PATHS["lstm"])
    print(f"✓ LSTM model loaded")
except Exception as e:
    print(f"✗ Failed to load LSTM model: {e}")

# Load dataset (for any potential future use)
print("\n=== Loading Dataset ===")
try:
    nft_data = pd.read_csv(PATHS["dataset"])
    if 'total_price' in nft_data.columns:
        nft_data['price_in_ether'] = nft_data['total_price'] / 1e18
        print("✓ Converted prices from wei to ether")
    else:
        nft_data['price_in_ether'] = nft_data.iloc[:, 0]
        print("⚠ Using first column as price values")
    print(f"✓ Loaded {len(nft_data)} records")
except Exception as e:
    print(f"✗ Failed to load dataset: {e}")
    nft_data = pd.DataFrame()

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_stream):
    """Process uploaded image for prediction"""
    try:
        img = Image.open(file_stream)
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        return img_array.reshape(1, *img_array.shape)
    except Exception as e:
        print(f"Image processing error: {e}")
        return None

def generate_nft_info(nft_name=None):
    """Generate mock NFT information (replace with real data from your models)"""
    return {
        'name': nft_name or 'Generated NFT #' + str(random.randint(1000, 9999)),
        'collection': 'AI Generated Collection',
        'rarity': random.choice(['Common', 'Uncommon', 'Rare', 'Epic', 'Legendary']),
        'traits': [
            {'trait': 'Background', 'value': random.choice(['Blue', 'Red', 'Purple'])},
            {'trait': 'Type', 'value': random.choice(['Animal', 'Robot', 'Alien'])},
            {'trait': 'Accessory', 'value': random.choice(['Hat', 'Glasses', 'None'])}
        ],
        'last_sale': f"{random.uniform(0.1, 2.0):.2f} ETH"
    }

def predict_from_image_features(img_features):
    """Predict price from image features - replace with actual model predictions"""
    return {
        'lr': random.uniform(0.5, 5.0),
        'knn': random.uniform(0.5, 5.0),
        'lstm': random.uniform(0.5, 5.0)
    }

# Routes
@app.route('/')
def home():
    return render_template(PATHS["index"])

@app.route('/predict_from_image', methods=['POST'])
def predict_from_image():
    if 'nft_image' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})
    
    file = request.files['nft_image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            # Process image
            img_data = file.read()
            img_features = preprocess_image(io.BytesIO(img_data))
            
            if img_features is None:
                return jsonify({'success': False, 'message': 'Image processing failed'})
            
            # Get predictions (replace with actual model predictions)
            predictions = predict_from_image_features(img_features)
            
            # Save the file
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(temp_path, 'wb') as f:
                f.write(img_data)
            
            # Generate NFT info and predictions
            nft_info = generate_nft_info()
            response = {
                'success': True,
                'lr': f"{predictions['lr']:.4f} ETH",
                'knn': f"{predictions['knn']:.4f} ETH",
                'lstm': f"{predictions['lstm']:.4f} ETH",
                'image_url': f"/{temp_path}",
                **nft_info
            }
            
            return jsonify(response)
        
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    
    return jsonify({'success': False, 'message': 'Invalid file type'})

if __name__ == '__main__':
    # Verify paths
    print("\n=== Path Verification ===")
    for name, path in PATHS.items():
        exists = os.path.exists(path) if not name.endswith('html') else True
        print(f"{'✓' if exists else '✗'} {name.ljust(20)}: {path}")
    
    print("\n=== Starting NFT Price Predictor ===")
    app.run(debug=True, port=5000)