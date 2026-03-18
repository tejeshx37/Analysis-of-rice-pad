import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from datetime import datetime
import sys

# ---------------------------
# Path Setup
# ---------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR.endswith("backend"):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

sys.path.append(ROOT_DIR)

try:
    from disease_info import disease_info
except ImportError:
    disease_info = {}

# ---------------------------
# Flask App Configuration
# ---------------------------
app = Flask(__name__, template_folder='../../templates', static_folder='../../static')
app.config['SECRET_KEY'] = 'rice-disease-detection-secret-key-2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(ROOT_DIR, 'rice_app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ---------------------------
# Database & Login Manager
# ---------------------------
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# ---------------------------
# Models
# ---------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    disease = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------------------------
# Create DB & Default User
# ---------------------------
with app.app_context():
    db.create_all()
    # Create a default admin user if none exists
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            password=generate_password_hash('admin123')
        )
        db.session.add(admin)
        db.session.commit()
        print("Default user created -> username: admin, password: admin123")

# ---------------------------
# PyTorch Model Setup
# ---------------------------
NUM_CLASSES = 20
MODEL_PATH = os.path.join(ROOT_DIR, "model", "best_resnet50_rice.pth")

CLASS_NAMES = [
    'Bacterial Blight', 'Bacterial Leaf Blight', 'Bacterial Streak', 'Bakanae',
    'Brown Spot', 'False Smut', 'Grassy Stunt Virus', 'Healthy', 'Hispa',
    'Leaf Blast', 'Leaf Smut', 'Leaf scald', 'Narrow Brown Spot', 'Neck Blast',
    'Ragged Stunt Virus', 'Rice False Smut', 'Sheath Blight', 'Sheath Rot',
    'Stem Rot', 'Tungro'
]

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

ai_model = models.resnet50(weights=None)
ai_model.fc = nn.Linear(ai_model.fc.in_features, NUM_CLASSES)

try:
    if os.path.exists(MODEL_PATH):
        ai_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

ai_model = ai_model.to(device)
ai_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------
# Auth Routes
# ---------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Username and password are required.', 'error')
        elif User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
        else:
            new_user = User(
                username=username,
                password=generate_password_hash(password)
            )
            db.session.add(new_user)
            db.session.commit()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('login.html', register_mode=True)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ---------------------------
# App Routes
# ---------------------------
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.content_type.startswith('image/'):
        return jsonify({'error': 'Invalid file type. Please upload a valid image.'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = ai_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

            predicted_class = CLASS_NAMES[predicted_idx.item()]
            confidence_score = round(confidence.item() * 100, 2)

        # Save prediction to database
        new_prediction = Prediction(
            disease=predicted_class,
            confidence=confidence_score,
            user_id=current_user.id
        )
        db.session.add(new_prediction)
        db.session.commit()

        info = disease_info.get(predicted_class, {})

        return jsonify({
            'disease': predicted_class,
            'confidence': confidence_score,
            'cause': info.get('cause', 'Information not available.'),
            'treatment': info.get('treatment', 'Information not available.'),
            'prevention': info.get('prevention', 'Information not available.')
        })

    except Exception as e:
        print(f"Prediction failed: {e}")
        return jsonify({'error': 'Prediction failure. Could not process image.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)