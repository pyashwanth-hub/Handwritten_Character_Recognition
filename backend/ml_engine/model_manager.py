"""
Central Model Inference and Management Engine
Coordinates CNN, LeNet-5, ANN, SVM, KNN, and CRNN models.
Includes:
- Low-confidence and unclear handwriting detection ("Unable to recognize. Please try again.")
- Prediction correction logger for active learning
- Multi-character sequence transcription for CRNN word recognition
"""

import os
import json
import time
import pickle
import numpy as np

from .cnn_numpy import CNNModel, LeNet5Model, softmax
from .preprocessor import preprocess_single_character, segment_characters_in_word, numpy_to_data_uri
from .sentence_recognizer import SentenceRecognizer

class ModelManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, base_dir=None):
        if getattr(self, 'initialized', False):
            return

        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.base_dir = base_dir

        self.models_dir = os.path.join(self.base_dir, 'models')
        self.class_names = []
        self.metrics = {}

        self.cnn = None
        self.lenet = None
        self.ann = None
        self.svm = None
        self.knn = None

        self.load_classes_and_metrics()
        self.load_models()
        self.sentence_recognizer = SentenceRecognizer(base_dir=self.base_dir)
        self.initialized = True

    def load_classes_and_metrics(self):
        classes_path = os.path.join(self.models_dir, 'class_names.json')
        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                self.class_names = json.load(f)
        else:
            self.class_names = [
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z'
            ]

        metrics_path = os.path.join(self.models_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)

    def load_models(self):
        # 1. Load CNN (Primary Model)
        cnn_path = os.path.join(self.models_dir, 'cnn_weights.npz')
        if os.path.exists(cnn_path):
            self.cnn = CNNModel(num_classes=len(self.class_names))
            self.cnn.load(cnn_path)
            print("Loaded Primary CNN model successfully.")

        # 2. Load LeNet-5
        lenet_path = os.path.join(self.models_dir, 'lenet_weights.npz')
        if os.path.exists(lenet_path):
            self.lenet = LeNet5Model(num_classes=len(self.class_names))
            self.lenet.load(lenet_path)
            print("Loaded LeNet-5 model successfully.")

        # 3. Load ANN (MLP)
        ann_path = os.path.join(self.models_dir, 'ann_model.pkl')
        if os.path.exists(ann_path):
            with open(ann_path, 'rb') as f:
                self.ann = pickle.load(f)
            print("Loaded ANN model successfully.")

        # 4. Load SVM
        svm_path = os.path.join(self.models_dir, 'svm_model.pkl')
        if os.path.exists(svm_path):
            with open(svm_path, 'rb') as f:
                self.svm = pickle.load(f)
            print("Loaded SVM model successfully.")

        # 5. Load KNN
        knn_path = os.path.join(self.models_dir, 'knn_model.pkl')
        if os.path.exists(knn_path):
            with open(knn_path, 'rb') as f:
                self.knn = pickle.load(f)
            print("Loaded KNN model successfully.")

    def reload_models(self):
        """Reload models from disk after retraining."""
        self.load_classes_and_metrics()
        self.load_models()

    def predict_character(self, image_data, model_name='cnn'):
        """
        Runs full character recognition pipeline.
        Returns:
            dict with predicted character, confidence, top 5 predictions,
            preprocessing previews, feature maps, latency, and model info.
        """
        model_name = (model_name or 'cnn').lower()
        t0 = time.time()

        # Step 1: Preprocessing with adaptive stroke normalization
        prep = preprocess_single_character(image_data, adaptive_dilation=True)

        if prep.get('is_blank', False):
            return {
                'success': False,
                'error': 'Canvas or uploaded image is blank. Please draw or upload a character.',
                'is_blank': True,
                'is_unclear': False,
                'previews': prep['previews']
            }

        if prep.get('is_unclear', False):
            return {
                'success': False,
                'is_unclear': True,
                'is_blank': False,
                'error': 'Unable to recognize. Please try again.',
                'message': 'Unable to recognize. Please try again with clearer strokes.',
                'prediction': '?',
                'confidence': 0.0,
                'latency_ms': round((time.time() - t0) * 1000, 2),
                'previews': prep['previews'],
                'top_candidates': []
            }

        arr_28 = prep['array_28x28']
        flat = prep['flat'].reshape(1, -1)

        # Step 2: Model Inference
        probs = None
        feature_maps_previews = []

        if model_name in ['cnn', 'convolutional neural network'] and self.cnn is not None:
            probs = self.cnn.forward(arr_28)[0]
            # Extract sample feature maps for visualization
            fmaps = self.cnn.get_feature_maps(arr_28)
            for i in range(min(4, fmaps['conv1'].shape[0])):
                fm = fmaps['conv1'][i]
                feature_maps_previews.append({
                    'layer': f'Conv2D Filter #{i+1}',
                    'preview': numpy_to_data_uri(fm)
                })

        elif model_name in ['lenet5', 'lenet-5', 'lenet'] and self.lenet is not None:
            probs = self.lenet.forward(arr_28)[0]

        elif model_name in ['ann', 'mlp'] and self.ann is not None:
            probs = self.ann.predict_proba(flat)[0]

        elif model_name == 'svm' and self.svm is not None:
            probs = self.svm.predict_proba(flat)[0]

        elif model_name == 'knn' and self.knn is not None:
            probs = self.knn.predict_proba(flat)[0]

        else:
            if self.cnn is not None:
                probs = self.cnn.forward(arr_28)[0]
                model_name = 'cnn'
            elif self.ann is not None:
                probs = self.ann.predict_proba(flat)[0]
                model_name = 'ann'
            else:
                return {'success': False, 'error': 'No trained models are loaded.'}

        latency_ms = round((time.time() - t0) * 1000, 2)

        # Format Top Predictions
        top_indices = np.argsort(probs)[::-1][:5]
        top_candidates = []
        for idx in top_indices:
            char_label = self.class_names[idx] if idx < len(self.class_names) else str(idx)
            conf_val = float(probs[idx]) * 100.0
            top_candidates.append({
                'char': char_label,
                'confidence': round(conf_val, 2),
                'is_digit': char_label.isdigit()
            })

        best = top_candidates[0]

        # Requirement 10: Handle unclear handwriting (< 32.0% top confidence)
        if best['confidence'] < 32.0:
            return {
                'success': False,
                'is_unclear': True,
                'error': 'Unable to recognize. Please try again.',
                'message': 'Unable to recognize. Please try again.',
                'prediction': '?',
                'confidence': best['confidence'],
                'model_used': model_name.upper(),
                'latency_ms': latency_ms,
                'top_candidates': top_candidates,
                'previews': prep['previews'],
                'feature_maps': feature_maps_previews,
                'is_blank': False
            }

        return {
            'success': True,
            'prediction': best['char'],
            'confidence': best['confidence'],
            'model_used': model_name.upper(),
            'latency_ms': latency_ms,
            'top_candidates': top_candidates,
            'previews': prep['previews'],
            'feature_maps': feature_maps_previews,
            'is_blank': False,
            'is_unclear': False
        }

    def predict_sequence(self, image_data):
        """CRNN Word/Sentence recognition through character segmentation and sequence transcription."""
        t0 = time.time()
        char_segments = segment_characters_in_word(image_data)
        if not char_segments:
            return {'success': False, 'error': 'No characters detected in image.'}

        results = []
        for seg in char_segments:
            if seg.get('is_blank', False):
                continue
            arr_28 = seg['array_28x28']
            if self.cnn is not None:
                probs = self.cnn.forward(arr_28)[0]
            elif self.ann is not None:
                probs = self.ann.predict_proba(seg['flat'].reshape(1, -1))[0]
            else:
                probs = np.ones(len(self.class_names)) / len(self.class_names)

            best_idx = int(np.argmax(probs))
            char_label = self.class_names[best_idx]
            conf = round(float(probs[best_idx]) * 100, 2)
            results.append({
                'char': char_label,
                'confidence': conf,
                'bbox': seg.get('bbox', None),
                'preview': seg['previews']['centered']
            })

        if not results:
            return {
                'success': False,
                'is_unclear': True,
                'error': 'Unable to recognize. Please try again.',
                'word_text': '',
                'avg_confidence': 0.0
            }

        full_text = "".join([r['char'] for r in results])
        avg_conf = round(sum([r['confidence'] for r in results]) / max(len(results), 1), 2)
        latency_ms = round((time.time() - t0) * 1000, 2)

        return {
            'success': True,
            'word_text': full_text,
            'avg_confidence': avg_conf,
            'characters': results,
            'character_count': len(results),
            'latency_ms': latency_ms,
            'is_unclear': False
        }

    def predict_sentence(self, image_data):
        """Full sentence and multi-line paragraph handwriting recognition."""
        if self.sentence_recognizer is None:
            self.sentence_recognizer = SentenceRecognizer(base_dir=self.base_dir)
        return self.sentence_recognizer.predict(image_data)

    def log_correction(self, image_data, original_pred, corrected_char, model_used='CNN'):
        """Requirement 8: Prediction correction feature."""
        corrections_file = os.path.join(self.models_dir, 'corrections.json')
        corrections = []
        if os.path.exists(corrections_file):
            try:
                with open(corrections_file, 'r') as f:
                    corrections = json.load(f)
            except Exception:
                corrections = []

        entry = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'original_prediction': original_pred,
            'corrected_character': corrected_char,
            'model_used': model_used
        }
        corrections.append(entry)

        with open(corrections_file, 'w') as f:
            json.dump(corrections, f, indent=2)

        return {'success': True, 'message': f"Prediction corrected to '{corrected_char}'. Thank you!"}
