"""
AI Handwritten Character Recognition - Backend REST API
Powered by Python, Flask, CNN, and Deep Learning.

Provides CORS-enabled REST endpoints for:
- Character prediction (/api/predict)
- Word sequence transcription (/api/predict-sequence)
- Model training analytics & metrics (/api/model-info)
- Sample character datasets (/api/sample-characters)
"""

import os
import json
import traceback
from flask import Flask, request, jsonify, send_from_directory
from ml_engine.model_manager import ModelManager

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), 'frontend')

app = Flask(__name__, static_folder=FRONTEND_DIR)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit

# Initialize Deep Learning Model Manager
model_manager = ModelManager(base_dir=BASE_DIR)

# --------------------------------------------------------------------------
# Cross-Origin Resource Sharing (CORS) Middleware
# Enables frontend on any port (e.g. port 3000, Live Server 5500) to communicate
# --------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# Handle CORS preflight requests
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    return '', 204

# --------------------------------------------------------------------------
# Optional Frontend Serving from Backend (Port 5000)
# --------------------------------------------------------------------------
@app.route('/')
def serve_index():
    if os.path.exists(os.path.join(FRONTEND_DIR, 'index.html')):
        return send_from_directory(FRONTEND_DIR, 'index.html')
    return jsonify({
        'status': 'online',
        'message': 'AI Handwritten Character Recognition REST API is running.',
        'endpoints': ['/api/predict', '/api/predict-sequence', '/api/model-info', '/api/sample-characters']
    })

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    return jsonify({'error': 'Resource not found'}), 404

# --------------------------------------------------------------------------
# REST API Endpoints
# --------------------------------------------------------------------------
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict single handwritten character.
    Payload: { "image": "data:image/png;base64,...", "model": "cnn" | "lenet5" | "ann" | "svm" | "knn" }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided.'}), 400

        image_data = data.get('image')
        model_name = data.get('model', 'cnn')

        result = model_manager.predict_character(image_data, model_name=model_name)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Inference error: {str(e)}'}), 500

@app.route('/api/predict-sequence', methods=['POST'])
def predict_sequence():
    """
    Predict word / sequence (CRNN mode).
    Payload: { "image": "data:image/png;base64,..." }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided.'}), 400

        image_data = data.get('image')
        result = model_manager.predict_sequence(image_data)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Sequence error: {str(e)}'}), 500

@app.route('/api/predict-sentence', methods=['POST'])
def predict_sentence():
    """
    Complete Handwritten Sentence & Multi-line Paragraph Recognition.
    Payload: { "image": "data:image/png;base64,..." }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided.'}), 400

        image_data = data.get('image')
        result = model_manager.predict_sentence(image_data)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Sentence recognition error: {str(e)}'}), 500

@app.route('/api/correct-prediction', methods=['POST'])
def correct_prediction():
    """
    Requirement 8: User Prediction Correction Feature.
    Payload: { "original": "P", "corrected": "E", "model": "CNN" }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        orig = data.get('original', '?')
        corr = data.get('corrected', '')
        model_name = data.get('model', 'CNN')
        img = data.get('image', None)

        if not corr:
            return jsonify({'success': False, 'error': 'No corrected character specified.'}), 400

        result = model_manager.log_correction(img, orig, corr, model_name)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Returns training history, per-model accuracy, and architecture details."""
    try:
        metrics_file = os.path.join(BASE_DIR, 'models', 'metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            return jsonify({'success': True, 'data': metrics_data})
        else:
            return jsonify({'success': False, 'error': 'Metrics not found. Models may still be training.'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/sample-characters', methods=['GET'])
def sample_characters():
    """Returns dataset samples for testing."""
    samples = [
        {'char': 'A', 'category': 'letter'},
        {'char': 'B', 'category': 'letter'},
        {'char': 'C', 'category': 'letter'},
        {'char': 'G', 'category': 'letter'},
        {'char': 'R', 'category': 'letter'},
        {'char': '3', 'category': 'digit'},
        {'char': '7', 'category': 'digit'},
        {'char': '8', 'category': 'digit'},
        {'char': '9', 'category': 'digit'},
        {'char': '0', 'category': 'digit'}
    ]
    return jsonify({'success': True, 'samples': samples})

if __name__ == '__main__':
    print("=" * 60)
    print("AI Handwritten Character Recognition - Backend REST API")
    print("Running on http://127.0.0.1:5000")
    print("CORS Enabled: Accessible from any frontend port")
    print("=" * 60)
    app.run(host='127.0.0.1', port=5000, debug=True)
