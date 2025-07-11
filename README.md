🧾 Counterfeit Currency Detection Using Image Processing
This project implements a simple image-based counterfeit currency detection system using computer vision and machine learning. It extracts features from images of currency notes and trains a machine learning model to distinguish between real and fake notes.

📂 Project Structure
php
Copy
Edit
.
├── extract_features.py       # Feature extraction using histogram of grayscale images
├── train_model.py            # Model training using Random Forest Classifier
├── dataset/
│   ├── real/                 # Folder containing real currency images
│   └── fake/                 # Folder containing fake currency images
├── model.pkl                 # Saved machine learning model (after training)
🛠️ Requirements
Python 3.x

OpenCV (cv2)

NumPy

scikit-learn

joblib

Install dependencies via pip:

bash
Copy
Edit
pip install opencv-python numpy scikit-learn joblib
📸 Feature Extraction
extract_features.py loads each currency image, converts it to grayscale, resizes it, and computes a normalized histogram of pixel intensities. This histogram serves as the feature vector used for training.

python
Copy
Edit
from extract_features import extract_features

features = extract_features('path/to/image.jpg')
🧠 Model Training
train_model.py:

Loads images from dataset/real and dataset/fake

Extracts features using the extract_features() function

Trains a RandomForestClassifier from scikit-learn

Saves the trained model as model.pkl

To train the model:

bash
Copy
Edit
python train_model.py
🧪 Example Dataset Structure
go
Copy
Edit
dataset/
├── real/
│   ├── note1.jpg
│   └── note2.jpg
└── fake/
    ├── noteA.jpg
    └── noteB.jpg
📦 Output
A trained model will be saved as:

Copy
Edit
model.pkl
This model can be used later to classify new currency images as real or fake based on the histogram features.

📈 Future Improvements
Use more advanced features (e.g., texture, ORB, SIFT)

Apply deep learning with CNNs for better accuracy

Add GUI or web interface for user interaction# Fake-Currency-Detection
