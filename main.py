import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TESSDATA_PREFIX'] = './tessdata'

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler
import string
import re
from PIL import Image, ImageOps
from datetime import datetime
from werkzeug.utils import secure_filename
import CharacterSegmentation as cs

app = Flask(__name__)
CORS(app)

# OCR configurations
UPLOAD_FOLDER = '../'
MODEL_PATH = 'haistudocr.h5'
SEGMENTED_DIR = './segmented/'
MAPPING_PATH = './emnist/processed-mapping.csv'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables
ocr_model = None
question_classifier_model = None
preprocessor = None
experts_df = None
expert_similarity_df = None
code2char = None

class DataPreprocessor:
    def __init__(self):
        self.features = None
        self.scaler = StandardScaler()
        
    def extract_text_features(self, text):
        # Lowercase the text
        text = str(text).lower()
        
        # Inisialisasi vektor fitur dengan zeros
        feature_vector = np.zeros(502)
        
        # 1. Basic text features (8 fitur pertama)
        char_count = len(text)
        word_count = len(text.split())
        punct_count = sum([1 for char in text if char in string.punctuation])
        number_count = sum([1 for char in text if char.isdigit()])
        words = text.split()
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        special_char_count = sum([1 for char in text if not char.isalnum() and char not in string.punctuation])
        uppercase_count = sum([1 for char in str(text) if char.isupper()])
        sentence_count = len(re.split(r'[.!?]+', text))
        
        feature_vector[0] = char_count
        feature_vector[1] = word_count
        feature_vector[2] = punct_count
        feature_vector[3] = number_count
        feature_vector[4] = avg_word_length
        feature_vector[5] = special_char_count
        feature_vector[6] = uppercase_count
        feature_vector[7] = sentence_count
        
        # 2. Character level features
        char_to_index = {chr(i): i-8 for i in range(32, 127)}
        for i, char in enumerate(text):
            if char in char_to_index and char_to_index[char] < 494:
                feature_vector[8 + char_to_index[char]] = 1
        
        return feature_vector
        
    def preprocess_new_data(self, question_data):
        features = self.extract_text_features(question_data['Problem'])
        processed_data = pd.DataFrame([features])
        
        if hasattr(self, 'scaler') and hasattr(self.scaler, 'mean_'):
            processed_data = pd.DataFrame(
                self.scaler.transform(processed_data),
                columns=processed_data.columns
            )
        
        return processed_data

class QuestionClassifier:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.difficulty_labels = ['Easy', 'Medium', 'Hard']
        
    def predict(self, question_text):
        try:
            question_data = {
                'Problem': question_text,
            }
            
            processed_data = self.preprocessor.preprocess_new_data(question_data)
            prediction = self.model.predict(processed_data)
            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction))
            
            return {
                'difficulty': self.difficulty_labels[predicted_class],
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}

def load_models_and_data():
    global ocr_model, question_classifier_model, preprocessor, experts_df, expert_similarity_df, code2char
    try:
        # 1. Load OCR model and mapping
        print("Loading OCR model and mapping...")
        ocr_model = tf.keras.models.load_model(MODEL_PATH)
        mapping_df = pd.read_csv(MAPPING_PATH)
        code2char = {row['id']: row['char'] for _, row in mapping_df.iterrows()}
        
        # Create necessary directories
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(SEGMENTED_DIR, exist_ok=True)
        
        # 2. Initialize preprocessor first
        print("Initializing preprocessor...")
        preprocessor = DataPreprocessor()
        
        # 3. Load question classifier model
        print("Loading question classifier model...")
        question_classifier_model = tf.keras.models.load_model('question_classifier_model.keras')
        
        # 4. Load expert recommendation data
        print("Loading expert recommendation data...")
        experts_df = pd.read_excel('expert.xlsx')
        with h5py.File('expert_similarity_model.h5', 'r') as file:
            similarity_data = file['similarity'][:]
            index_data = file['index'][:]
            expert_similarity_df = pd.DataFrame(similarity_data, 
                                             index=index_data, 
                                             columns=index_data)
        
        print("All models and data loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models and data: {str(e)}")
        return False

def process_image(filepath):
    """OCR Pipeline"""
    # 1. Character Segmentation
    cs.image_segmentation(filepath)

    # 2. Load segmented images
    segmented_images = []
    for file in os.listdir(SEGMENTED_DIR):
        if file.endswith('.jpg'):
            segmented_images.append(os.path.join(SEGMENTED_DIR, file))

    # 3. Preprocess segmented images
    X_data = []
    for img_path in segmented_images:
        img = Image.open(img_path).resize((28, 28))
        inv_img = ImageOps.invert(img)
        flatten = np.array(inv_img).flatten() / 255
        flatten = np.where(flatten > 0.5, 1, 0)
        X_data.append(flatten)

    X_data = np.array(X_data).reshape(-1, 28, 28, 1)

    # 4. Predict and convert to text
    predictions = ocr_model.predict(X_data)
    predicted_indices = np.argmax(predictions, axis=1)
    recognized_text = ''.join([code2char[idx] for idx in predicted_indices])
    return recognized_text

def get_expert_recommendations(top_n=3):
    """Get expert recommendations from Excel file"""
    try:
        if experts_df is None:
            print("Error: experts_df is None")
            return []
        
        # Ambil daftar expertise dari Excel
        # Asumsikan ada kolom 'expertise' di Excel
        expertise_list = experts_df['expertise'].tolist()
        
        # Ambil top_n expertise
        if expertise_list:
            # Ambil unique values untuk menghindari duplikat
            unique_expertise = list(set(expertise_list))
            # Ambil sejumlah top_n atau semua jika jumlahnya kurang dari top_n
            recommendations = unique_expertise[:top_n]
            print(f"Returning {len(recommendations)} recommendations: {recommendations}")
            return recommendations
            
        return []
        
    except Exception as e:
        print(f"Error in get_expert_recommendations: {str(e)}")
        return []

@app.route('/process', methods=['POST'])
def process():
    """Endpoint untuk menerima input teks dan gambar sekaligus"""
    try:
        # Inisialisasi variabel untuk menyimpan teks
        recognized_text = ""
        
        # Cek jika ada input teks
        text_input = request.form.get('text', '')
        if text_input:
            recognized_text = text_input
        
        # Cek jika ada input gambar
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                ocr_text = process_image(filepath)
                # Gabungkan dengan teks input jika ada
                recognized_text = ocr_text if not recognized_text else f"{recognized_text}\n{ocr_text}"
        
        # Jika tidak ada input sama sekali
        if not recognized_text:
            return jsonify({'error': 'No input provided (text or image)'}), 400
            
        # Proses klasifikasi
        classifier = QuestionClassifier(question_classifier_model, preprocessor)
        classification_result = classifier.predict(recognized_text)
        
        # Dapatkan rekomendasi
        recommendations = get_expert_recommendations(top_n=3)
        
        # Return hasil
        return jsonify({
            'text': recognized_text,
            'classification': classification_result,
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(f"Process error: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    except Exception as e:
        print(f"Process error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Tambahkan route untuk mengecek status preprocessor
@app.route('/check-preprocessor', methods=['GET'])
def check_preprocessor():
    return jsonify({
        'preprocessor_status': preprocessor is not None,
        'preprocessor_type': str(type(preprocessor)) if preprocessor is not None else None
    })

@app.route('/check-experts', methods=['GET'])
def check_experts():
    try:
        if experts_df is None:
            return jsonify({'status': 'error', 'message': 'Experts data not loaded'}), 500
        
        return jsonify({
            'status': 'success',
            'data': {
                'total_experts': len(experts_df),
                'difficulty_levels': experts_df['difficulty_level'].unique().tolist(),
                'columns': experts_df.columns.tolist(),
                'first_row': experts_df.iloc[0].to_dict() if len(experts_df) > 0 else None
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'ocr_model_loaded': ocr_model is not None,
        'classifier_model_loaded': question_classifier_model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'experts_data_loaded': experts_df is not None
    })

# Initialize on startup
load_models_and_data()

if __name__ == '__main__':
    app.run(debug=True)
