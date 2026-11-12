"""
Comprehensive Verification and Test Suite
Tests model loading, single character prediction, CRNN word sequence recognition,
preprocessing pipeline, and Flask API endpoints.
"""

import sys
import os
import unittest
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ml_engine.model_manager import ModelManager
from ml_engine.preprocessor import preprocess_single_character, pil_to_data_uri
from app import app

def generate_test_image(char='A'):
    img = Image.new('L', (140, 140), 0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 90)
    except Exception:
        font = ImageFont.load_default()
    draw.text((35, 15), char, fill=255, font=font)
    return img

class TestHandwrittenCharacterRecognition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n========================================================")
        print("RUNNING HANDWRITTEN CHARACTER RECOGNITION TEST SUITE")
        print("========================================================")
        cls.manager = ModelManager(base_dir=BASE_DIR)
        cls.test_char_A = pil_to_data_uri(generate_test_image('A'))
        cls.test_char_7 = pil_to_data_uri(generate_test_image('7'))
        cls.client = app.test_client()

    def test_01_models_loaded(self):
        print("\n[Test 1] Verifying all models loaded in ModelManager...")
        self.assertIsNotNone(self.manager.cnn, "CNN model failed to load")
        self.assertIsNotNone(self.manager.lenet, "LeNet-5 model failed to load")
        self.assertIsNotNone(self.manager.ann, "ANN model failed to load")
        self.assertIsNotNone(self.manager.svm, "SVM model failed to load")
        self.assertIsNotNone(self.manager.knn, "KNN model failed to load")
        self.assertEqual(len(self.manager.class_names), 36)
        print("  --> OK: All 5 models loaded successfully with 36 classes (0-9, A-Z)!")

    def test_02_cnn_prediction(self):
        print("\n[Test 2] Testing CNN Primary Model inference on character 'A'...")
        res = self.manager.predict_character(self.test_char_A, model_name='cnn')
        self.assertTrue(res['success'])
        self.assertEqual(res['prediction'], 'A')
        self.assertGreater(res['confidence'], 70.0)
        self.assertEqual(len(res['top_candidates']), 5)
        self.assertGreater(len(res['feature_maps']), 0)
        print(f"  --> OK: CNN predicted '{res['prediction']}' with {res['confidence']}% confidence in {res['latency_ms']}ms!")

    def test_03_digit_prediction(self):
        print("\n[Test 3] Testing prediction on handwritten digit '3'...")
        test_char_3 = pil_to_data_uri(generate_test_image('3'))
        res = self.manager.predict_character(test_char_3, model_name='cnn')
        self.assertTrue(res['success'])
        self.assertEqual(res['prediction'], '3')
        self.assertGreater(res['confidence'], 80.0)
        print(f"  --> OK: CNN predicted digit '{res['prediction']}' with {res['confidence']}% confidence in {res['latency_ms']}ms!")

    def test_04_all_models_inference(self):
        print("\n[Test 4] Testing inference across all 5 models (CNN, LeNet-5, ANN, SVM, KNN)...")
        for m in ['cnn', 'lenet5', 'ann', 'svm', 'knn']:
            res = self.manager.predict_character(self.test_char_A, model_name=m)
            self.assertTrue(res['success'], f"Model {m} failed")
            print(f"  --> Model [{m.upper()}]: Predicted '{res['prediction']}' ({res['confidence']}%) in {res['latency_ms']}ms")

    def test_05_word_sequence_crnn(self):
        print("\n[Test 5] Testing CRNN multi-character sequence transcription...")
        # Create a word image with "AI"
        word_img = Image.new('L', (280, 140), 0)
        draw = ImageDraw.Draw(word_img)
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 80)
        draw.text((30, 25), "A", fill=255, font=font)
        draw.text((150, 25), "I", fill=255, font=font)
        data_uri = pil_to_data_uri(word_img)

        res = self.manager.predict_sequence(data_uri)
        self.assertTrue(res['success'])
        self.assertEqual(res['character_count'], 2)
        print(f"  --> OK: CRNN decoded word '{res['word_text']}' with {res['avg_confidence']}% confidence in {res['latency_ms']}ms!")

    def test_06_flask_endpoints(self):
        print("\n[Test 6] Testing Flask REST API endpoints...")
        
        # 1. GET /
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)

        # 2. GET /api/model-info
        resp = self.client.get('/api/model-info')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['success'])
        self.assertIn('models', data['data'])

        # 3. POST /api/predict
        resp = self.client.post('/api/predict', json={'image': self.test_char_A, 'model': 'cnn'})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['prediction'], 'A')

        # 4. GET /api/sample-characters
        resp = self.client.get('/api/sample-characters')
        self.assertEqual(resp.status_code, 200)

        print("  --> OK: All Flask API endpoints responded with HTTP 200 OK!")

    def test_07_sentence_recognition(self):
        print("\n[Test 7] Testing Complete Handwritten Sentence Recognition (/api/predict-sentence)...")
        # Render target sentence: "Machine learning is an interesting technology."
        from PIL import ImageFont
        font_path = 'C:/Windows/Fonts/Inkfree.ttf'
        font = ImageFont.truetype(font_path, 24) if os.path.exists(font_path) else ImageFont.load_default()
        test_sent = "Machine learning is an interesting technology."
        img = Image.new('L', (600, 50), 0)
        d = ImageDraw.Draw(img)
        d.text((20, 10), test_sent, fill=255, font=font)
        sent_uri = pil_to_data_uri(img)

        resp = self.client.post('/api/predict-sentence', json={'image': sent_uri})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['full_text'], test_sent)
        self.assertGreaterEqual(data['confidence'], 75.0)
        self.assertEqual(data['word_count'], 6)
        print(f"  --> OK: Recognized sentence '{data['full_text']}' ({data['confidence']}%) with {data['word_count']} words in {data['latency_ms']}ms!")

if __name__ == '__main__':
    unittest.main()
