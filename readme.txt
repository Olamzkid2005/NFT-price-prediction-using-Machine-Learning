NFT Price Predictor - Machine Learning System 🖼️📈
A Flask-based web application that predicts NFT prices using Linear Regression, KNN, and LSTM models.


Table of Contents
How It Works

Features

Models Used

Installation

Usage

API Endpoints

Future Improvements

Contributing

How It Works 🛠️
This system predicts NFT prices using three machine learning models:

Linear Regression (Baseline) – Simple trend analysis.

K-Nearest Neighbors (KNN) – Price based on similar NFTs.

LSTM Neural Network – Time-series forecasting for historical trends.

Prediction Flow
User uploads an NFT image (or selects from dataset).

Backend processes the image (feature extraction).

Three models generate predictions:

Linear Regression → Fast, interpretable estimate.

KNN → Compares with similar NFTs.

LSTM → Analyzes price trends over time.

Results displayed in a clean UI with confidence scores.

Features ✨
✅ Multi-Model Predictions – Compare Linear Regression, KNN, and LSTM results.
✅ Image Upload – Supports JPG, PNG, GIF.
✅ Historical Price Trends – LSTM learns from past sales.
✅ Responsive UI – Works on desktop & mobile.
✅ OpenSea/Rarible Integration – Compare predictions with real market data.

Models Used 🤖
Model	Best For	Pros	Cons
Linear Regression	Baseline prediction	Fast, simple	Assumes linear trends
KNN	Similar NFTs	Works with small data	Slow for big datasets
LSTM	Time-series forecasting	Captures trends	Needs lots of data


Installation ⚙️

1. Clone the Repository
bash
git clone https://github.com/Olamzkid2005/NFT-price-prediction-using-Machine-Learning
cd nft-price-predictor

2. Install Dependencies
bash
pip install -r requirements.txt

3. Run the Flask App
bash
python app.py
➡️ Open http://localhost:5000 in your browser.

Usage 🖥️
Upload an NFT image (or use sample data).

View price predictions from all three models.

Compare with OpenSea/Rarible using provided links.

Demo Screenshot (Add a screenshot here later)

API Endpoints 🌐
Endpoint	Method	Description
/	GET	Homepage (UI)
/predict_from_image	POST	Upload NFT image for prediction
/predict_from_name	GET	Predict using NFT name (dataset)

Future Improvements 🚀
Add more models (Random Forest, XGBoost).

Improve image feature extraction (CNN-based).

Deploy on AWS/GCP for scalability.

Add user accounts to save prediction history.



License 📜
MIT © [OLAMIJULO DAVID]

Enjoy predicting NFT prices! 🎨🚀
