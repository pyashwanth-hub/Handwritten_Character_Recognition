"""
CRNN (Convolutional Recurrent Neural Network + CTC) for Handwritten Sentence & Word Recognition.
High-Performance NumPy + SciPy Deep Learning Engine:
- CNN Feature Extraction: 16 Conv2D filters with 4-zone anatomical pooling (Ascender, Midline, Baseline, Descender)
- Sequence Modeling: Bidirectional LSTM (BiLSTM) capturing forward and reverse temporal context
- Linear Projection + Softmax over vocabulary (Letters A-Z/a-z, Digits 0-9, Punctuation, Space, and CTC blank)
- CTC Greedy Decoding with repeat token collapsing, blank removal, and word boundary alignment
"""

import os
import numpy as np
from scipy import ndimage

# Full OCR Vocabulary
VOCAB_CHARS = [
    '<blank>',  # Index 0: CTC blank
    ' ',        # Index 1: Word space
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '.', ',', '!', '?', "'", '"', '-', ':', ';'
]

CHAR2IDX = {ch: i for i, ch in enumerate(VOCAB_CHARS)}
IDX2CHAR = {i: ch for i, ch in enumerate(VOCAB_CHARS)}
VOCAB_SIZE = len(VOCAB_CHARS)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class LSTMCell:
    def __init__(self, in_features, hidden_size):
        self.in_features = in_features
        self.hidden_size = hidden_size
        scale = np.sqrt(2.0 / (in_features + hidden_size))
        self.W = (np.random.randn(4 * hidden_size, in_features) * scale).astype(np.float32)
        self.U = (np.random.randn(4 * hidden_size, hidden_size) * scale).astype(np.float32)
        self.b = np.zeros(4 * hidden_size, dtype=np.float32)
        self.b[hidden_size:2*hidden_size] = 1.0

    def step(self, x_t, h_prev, c_prev):
        H = self.hidden_size
        gates = np.dot(self.W, x_t) + np.dot(self.U, h_prev) + self.b
        i = sigmoid(gates[0:H])
        f = sigmoid(gates[H:2*H])
        o = sigmoid(gates[2*H:3*H])
        g = np.tanh(gates[3*H:4*H])
        c_t = f * c_prev + i * g
        h_t = o * np.tanh(c_t)
        return h_t, c_t


class BiLSTM:
    def __init__(self, in_features, hidden_size):
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.forward_cell = LSTMCell(in_features, hidden_size)
        self.backward_cell = LSTMCell(in_features, hidden_size)

    def forward(self, seq):
        T = seq.shape[0]
        H = self.hidden_size

        h_f = np.zeros(H, dtype=np.float32)
        c_f = np.zeros(H, dtype=np.float32)
        fwd_states = np.zeros((T, H), dtype=np.float32)
        for t in range(T):
            h_f, c_f = self.forward_cell.step(seq[t], h_f, c_f)
            fwd_states[t] = h_f

        h_b = np.zeros(H, dtype=np.float32)
        c_b = np.zeros(H, dtype=np.float32)
        bwd_states = np.zeros((T, H), dtype=np.float32)
        for t in reversed(range(T)):
            h_b, c_b = self.backward_cell.step(seq[t], h_b, c_b)
            bwd_states[t] = h_b

        return np.concatenate([fwd_states, bwd_states], axis=-1)


class CRNNModel:
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_size=64):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # 16 Conv2D filters (3x3)
        self.W_conv = (np.random.randn(16, 3, 3) * np.sqrt(2.0 / 9)).astype(np.float32)
        self.b_conv = np.zeros(16, dtype=np.float32)

        # BiLSTM: 64 in -> 2*64 = 128 out
        self.bilstm = BiLSTM(in_features=64, hidden_size=hidden_size)

        # Output projection: 128 -> vocab_size
        self.W_proj = (np.random.randn(vocab_size, 2 * hidden_size) * np.sqrt(2.0 / (2 * hidden_size))).astype(np.float32)
        self.b_proj = np.zeros(vocab_size, dtype=np.float32)

    def extract_cnn_features(self, x):
        H, W = x.shape
        if W % 4 != 0:
            pad_w = 4 - (W % 4)
            x = np.pad(x, ((0, 0), (0, pad_w)), mode='constant')
            W = x.shape[1]

        if H != 32:
            from scipy.ndimage import zoom
            x = zoom(x, (32.0 / H, 1.0), order=1)

        out_c = np.zeros((16, 32, W), dtype=np.float32)
        for f in range(16):
            conv_f = ndimage.convolve(x, self.W_conv[f], mode='constant', cval=0.0) + self.b_conv[f]
            out_c[f] = np.maximum(0.0, conv_f)

        T = W // 4
        zones = out_c.reshape(16, 4, 8, T, 4).max(axis=(2, 4))
        seq_features = zones.reshape(64, T).T
        return seq_features

    def forward(self, line_img):
        features = self.extract_cnn_features(line_img)
        lstm_out = self.bilstm.forward(features)
        logits = np.dot(lstm_out, self.W_proj.T) + self.b_proj
        probs = softmax(logits, axis=-1)
        return probs

    def decode_ctc(self, probs, orig_width=None):
        T, K = probs.shape
        raw_indices = np.argmax(probs, axis=-1)
        confidences = np.max(probs, axis=-1)

        collapsed_tokens = []
        for t in range(T):
            idx = raw_indices[t]
            if idx == 0:
                continue
            if t > 0 and idx == raw_indices[t - 1]:
                continue
            char = IDX2CHAR.get(idx, '')
            if char:
                collapsed_tokens.append({
                    'char': char,
                    'idx': idx,
                    't': t,
                    'confidence': float(confidences[t]) * 100.0
                })

        if not collapsed_tokens:
            return {
                'text': '',
                'confidence': 0.0,
                'words': []
            }

        full_text = "".join([tok['char'] for tok in collapsed_tokens]).strip()

        words = []
        current_word_chars = []
        current_word_confs = []
        current_t_start = 0

        scale_x = (orig_width / float(T)) if orig_width is not None and T > 0 else 4.0

        for tok in collapsed_tokens:
            if tok['char'] == ' ':
                if current_word_chars:
                    w_str = "".join(current_word_chars)
                    w_conf = round(float(np.mean(current_word_confs)), 2)
                    x1 = int(current_t_start * scale_x)
                    x2 = int(tok['t'] * scale_x)
                    words.append({
                        'word': w_str,
                        'confidence': w_conf,
                        'bbox': [x1, 0, x2, 32]
                    })
                    current_word_chars = []
                    current_word_confs = []
            else:
                if not current_word_chars:
                    current_t_start = tok['t']
                current_word_chars.append(tok['char'])
                current_word_confs.append(tok['confidence'])

        if current_word_chars:
            w_str = "".join(current_word_chars)
            w_conf = round(float(np.mean(current_word_confs)), 2)
            x1 = int(current_t_start * scale_x)
            x2 = int(min(T * scale_x, (collapsed_tokens[-1]['t'] + 1) * scale_x))
            words.append({
                'word': w_str,
                'confidence': w_conf,
                'bbox': [x1, 0, x2, 32]
            })

        avg_conf = round(float(np.mean([t['confidence'] for t in collapsed_tokens])), 2)

        return {
            'text': full_text,
            'confidence': avg_conf,
            'words': words
        }

    def save(self, filepath):
        np.savez_compressed(
            filepath,
            W_conv=self.W_conv,
            b_conv=self.b_conv,
            fwd_W=self.bilstm.forward_cell.W,
            fwd_U=self.bilstm.forward_cell.U,
            fwd_b=self.bilstm.forward_cell.b,
            bwd_W=self.bilstm.backward_cell.W,
            bwd_U=self.bilstm.backward_cell.U,
            bwd_b=self.bilstm.backward_cell.b,
            W_proj=self.W_proj,
            b_proj=self.b_proj
        )

    def load(self, filepath):
        data = np.load(filepath)
        self.W_conv = data['W_conv']
        self.b_conv = data['b_conv']
        self.bilstm.forward_cell.W = data['fwd_W']
        self.bilstm.forward_cell.U = data['fwd_U']
        self.bilstm.forward_cell.b = data['fwd_b']
        self.bilstm.backward_cell.W = data['bwd_W']
        self.bilstm.backward_cell.U = data['bwd_U']
        self.bilstm.backward_cell.b = data['bwd_b']
        self.W_proj = data['W_proj']
        self.b_proj = data['b_proj']
