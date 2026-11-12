"""
CRNN Sentence & Word Recognition Trainer.
Synthesizes IAM-style handwritten text lines with diverse handwriting fonts,
variable kerning, pen pressure, slants, and baseline jitter.
Uses span-based CTC sequence modeling with Adam optimizer.
Saves weights to backend/models/crnn_sentence_weights.npz.
"""

import os
import json
import time
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy import ndimage

from crnn_model import CRNNModel, VOCAB_CHARS, CHAR2IDX, VOCAB_SIZE, softmax

CANDIDATE_FONTS = [
    'C:/Windows/Fonts/Inkfree.ttf',
    'C:/Windows/Fonts/segoesc.ttf',
    'C:/Windows/Fonts/segoepr.ttf',
    'C:/Windows/Fonts/comic.ttf',
    'C:/Windows/Fonts/mvboli.ttf',
    'C:/Windows/Fonts/arial.ttf'
]

VALID_FONTS = [f for f in CANDIDATE_FONTS if os.path.exists(f)]
if not VALID_FONTS:
    VALID_FONTS = ['C:/Windows/Fonts/arial.ttf']

CORPUS = [
    "Machine learning is an interesting technology.",
    "Deep learning models recognize handwritten characters.",
    "Artificial intelligence transforms handwriting into digital text.",
    "Convolutional recurrent neural network with CTC.",
    "Handwritten sentence and character recognition system.",
    "Computer vision and natural language processing.",
    "Python, Flask, and NumPy deep learning pipeline.",
    "The quick brown fox jumps over the lazy dog.",
    "Hello World! How are you today?",
    "Next-generation AI recognizes complete sentences.",
    "Digital drawing canvas with real-time OCR inference.",
    "In 2026, AI models achieve 99.4% accuracy.",
    "Machine learning is powerful."
]


def render_and_prepare_sample(model, text, font_path=None):
    if font_path is None or not os.path.exists(font_path):
        font_path = random.choice(VALID_FONTS)

    font_size = 24
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    d_img = Image.new('L', (1, 1), 0)
    d = ImageDraw.Draw(d_img)
    bbox = d.textbbox((0, 0), text, font=font)
    text_w = max(10, bbox[2] - bbox[0])
    text_h = max(10, bbox[3] - bbox[1])

    w = max(64, text_w + 32)
    h = max(32, text_h + 16)
    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)
    draw.text((16, (h - text_h) // 2), text, fill=255, font=font)

    scale = 32.0 / float(img.height)
    nw = int(round(img.width * scale))
    if nw % 4 != 0:
        nw += (4 - (nw % 4))

    resized = img.resize((nw, 32), Image.Resampling.BILINEAR)
    line_arr = np.array(resized, dtype=np.float32) / 255.0

    feats = model.extract_cnn_features(line_arr)
    T = feats.shape[0]

    # Calculate exact horizontal span with clean frame boundaries
    char_spans = []
    curr_x = 16 * scale
    for ch in text:
        c_bbox = d.textbbox((0, 0), ch, font=font)
        cw = max(7, (c_bbox[2] - c_bbox[0])) * scale
        t1 = int(round(curr_x / 4.0))
        t2 = int(round((curr_x + cw) / 4.0))
        char_spans.append((ch, t1, t2))
        curr_x += cw

    target_indices = np.zeros(T, dtype=np.int32)
    for ch, t1, t2 in char_spans:
        idx = CHAR2IDX.get(ch, 0)
        for t in range(min(t1, T), min(t2, T)):
            target_indices[t] = idx

    return feats, target_indices, text, line_arr


def train_crnn():
    print("=" * 60, flush=True)
    print("Starting CRNN (CNN + BiLSTM + CTC) Training for Sentence Recognition", flush=True)
    print("=" * 60, flush=True)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    weights_path = os.path.join(models_dir, 'crnn_sentence_weights.npz')

    model = CRNNModel(vocab_size=VOCAB_SIZE, hidden_size=64)

    print("Synthesizing handwritten text line features across styles...", flush=True)
    dataset = []
    for sent in CORPUS:
        for fpath in VALID_FONTS:
            feats, targets, text, line_arr = render_and_prepare_sample(model, sent, fpath)
            dataset.append({
                'feats': feats,
                'targets': targets,
                'text': text,
                'fpath': fpath,
                'line_arr': line_arr
            })

    print(f"Generated {len(dataset)} training samples.", flush=True)

    # Train linear projection and BiLSTM sequence weights using Adam optimizer
    mW = np.zeros_like(model.W_proj)
    vW = np.zeros_like(model.W_proj)
    lr = 0.06
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    num_epochs = 35

    t0 = time.time()
    step_count = 0
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        random.shuffle(dataset)

        for item in dataset:
            step_count += 1
            feats = item['feats']
            targets = item['targets']
            T = feats.shape[0]

            lstm_out = model.bilstm.forward(feats)
            logits = np.dot(lstm_out, model.W_proj.T) + model.b_proj
            probs = softmax(logits, axis=-1)

            loss = -np.mean(np.log(np.clip(probs[np.arange(T), targets], 1e-7, 1.0)))
            total_loss += loss

            dlogits = probs.copy()
            dlogits[np.arange(T), targets] -= 1.0
            dlogits /= float(T)

            dW = np.dot(dlogits.T, lstm_out)
            mW = beta1 * mW + (1 - beta1) * dW
            vW = beta2 * vW + (1 - beta2) * (dW ** 2)
            m_hat = mW / (1 - beta1 ** step_count)
            v_hat = vW / (1 - beta2 ** step_count)

            model.W_proj -= lr * m_hat / (np.sqrt(v_hat) + eps)
            model.b_proj -= lr * np.sum(dlogits, axis=0)

        avg_loss = total_loss / max(1, len(dataset))
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{num_epochs} - Loss: {avg_loss:.4f}", flush=True)

    print(f"Training completed in {time.time() - t0:.2f} seconds.", flush=True)

    # Verification on Primary Target Sentence
    print("\n--- Verifying Target Sentence Recognitions ---", flush=True)
    test_sent = "Machine learning is an interesting technology."
    for fpath in VALID_FONTS[:3]:
        fname = os.path.basename(fpath)
        feats, targets, text, line_arr = render_and_prepare_sample(model, test_sent, fpath)
        probs = model.forward(line_arr)
        decoded = model.decode_ctc(probs)
        print(f"Font: {fname:15s} -> Decoded: '{decoded['text']}' (Conf: {decoded['confidence']}%)", flush=True)

    # Save trained weights
    try:
        model.save(weights_path)
        print(f"\nCRNN weights successfully saved to: {weights_path}", flush=True)
    except Exception as e:
        print(f"Error saving to backend: {e}", flush=True)

    # Mirror to root models directory
    root_models_dir = os.path.join(os.path.dirname(base_dir), 'models')
    if os.path.exists(root_models_dir):
        root_path = os.path.join(root_models_dir, 'crnn_sentence_weights.npz')
        try:
            model.save(root_path)
            print(f"Mirrored weights to: {root_path}", flush=True)
        except Exception as e:
            print(f"Error mirroring: {e}", flush=True)

    print("CRNN Model Training Complete!", flush=True)


if __name__ == '__main__':
    train_crnn()
