"""
Pure NumPy Deep Learning Convolutional Neural Network Engine
Implements:
- Conv2D (with fast im2col vectorized operations)
- MaxPool2D (2x2 with stride 2)
- ReLU activation
- Flatten
- Dense (Fully Connected)
- Softmax probability layer
- Adam optimizer and Cross-Entropy loss for training
- LeNet-5 and Modern CNN architecture configurations
"""

import numpy as np

def im2col_indices(x, field_height=3, field_width=3, padding=1, stride=1):
    """An implementation of im2col based on indexing for maximum CPU performance."""
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def get_im2col_indices(x_shape, field_height=3, field_width=3, padding=1, stride=1):
    N, C, H, W = x_shape
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return (k.astype(int), i.astype(int), j.astype(int))

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """Fast col2im based on indexing."""
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # He / Xavier normal initialization
        std = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        self.b = np.zeros((out_channels, 1), dtype=np.float32)

        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        out_h = int((H + 2 * self.padding - self.kernel_size) / self.stride + 1)
        out_w = int((W + 2 * self.padding - self.kernel_size) / self.stride + 1)

        self.x_col = im2col_indices(x, self.kernel_size, self.kernel_size, padding=self.padding, stride=self.stride)
        W_row = self.W.reshape(self.out_channels, -1)

        out = np.dot(W_row, self.x_col) + self.b
        out = out.reshape(self.out_channels, out_h, out_w, N)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        N, C, H, W = self.x.shape
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        dW = np.dot(dout_reshaped, self.x_col.T).reshape(self.W.shape)
        db = np.sum(dout_reshaped, axis=1, keepdims=True)

        W_row = self.W.reshape(self.out_channels, -1)
        dx_col = np.dot(W_row.T, dout_reshaped)
        dx = col2im_indices(dx_col, self.x.shape, self.kernel_size, self.kernel_size, padding=self.padding, stride=self.stride)
        return dx, dW, db

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        h_out = int((H - self.pool_size) / self.stride + 1)
        w_out = int((W - self.pool_size) / self.stride + 1)

        x_reshaped = x.reshape(N * C, 1, H, W)
        self.x_col = im2col_indices(x_reshaped, self.pool_size, self.pool_size, padding=0, stride=self.stride)

        self.max_idx = np.argmax(self.x_col, axis=0)
        out = self.x_col[self.max_idx, np.arange(self.max_idx.size)]
        out = out.reshape(h_out, w_out, N, C)
        out = out.transpose(2, 3, 0, 1)
        return out

    def backward(self, dout):
        N, C, H, W = self.x.shape
        dx_col = np.zeros_like(self.x_col)
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()
        dx_col[self.max_idx, np.arange(self.max_idx.size)] = dout_flat

        x_reshaped = self.x.reshape(N * C, 1, H, W)
        dx = col2im_indices(dx_col, x_reshaped.shape, self.pool_size, self.pool_size, padding=0, stride=self.stride)
        return dx.reshape(N, C, H, W)

class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        return dout * (self.x > 0)

class Flatten:
    def forward(self, x):
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.orig_shape)

class Dense:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        std = np.sqrt(2.0 / in_features)
        self.W = (np.random.randn(in_features, out_features).astype(np.float32) * std)
        self.b = np.zeros((1, out_features), dtype=np.float32)

        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0, keepdims=True)
        return dx, dW, db

def softmax(x):
    """Numerically stable softmax for 2D matrix (batch_size, num_classes)."""
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class CNNModel:
    """Modern Deep CNN architecture for character recognition."""
    def __init__(self, num_classes=36):
        self.num_classes = num_classes
        # Layers:
        # Conv1 (1 -> 16, 3x3) -> ReLU -> MaxPool (28x28 -> 14x14)
        # Conv2 (16 -> 32, 3x3) -> ReLU -> MaxPool (14x14 -> 7x7)
        # Flatten (7*7*32 = 1568)
        # Dense (1568 -> 128) -> ReLU
        # Dense (128 -> num_classes)
        self.conv1 = Conv2D(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2, 2)

        self.conv2 = Conv2D(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2, 2)

        self.flatten = Flatten()
        self.fc1 = Dense(7 * 7 * 32, 128)
        self.relu3 = ReLU()
        self.fc2 = Dense(128, num_classes)

        self.params = [self.conv1, self.conv2, self.fc1, self.fc2]

    def forward(self, x):
        """Returns softmax probabilities of shape (N, num_classes)."""
        # Ensure N, C, H, W
        if x.ndim == 3:
            x = x[:, np.newaxis, :, :]
        elif x.ndim == 2:
            x = x[np.newaxis, np.newaxis, :, :]

        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)

        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)

        out = self.flatten.forward(out)
        out = self.fc1.forward(out)
        out = self.relu3.forward(out)
        logits = self.fc2.forward(out)

        probs = softmax(logits)
        return probs

    def get_feature_maps(self, x):
        """Extract intermediate activations for UI visualization."""
        if x.ndim == 3:
            x = x[:, np.newaxis, :, :]
        elif x.ndim == 2:
            x = x[np.newaxis, np.newaxis, :, :]

        c1 = self.conv1.forward(x)
        r1 = self.relu1.forward(c1)
        p1 = self.pool1.forward(r1)

        c2 = self.conv2.forward(p1)
        r2 = self.relu2.forward(c2)
        p2 = self.pool2.forward(r2)

        return {
            'conv1': c1[0], # (16, 28, 28)
            'pool1': p1[0], # (16, 14, 14)
            'conv2': c2[0], # (32, 14, 14)
            'pool2': p2[0], # (32, 7, 7)
        }

    def save(self, filepath):
        """Save weights to .npz file."""
        np.savez_compressed(
            filepath,
            conv1_W=self.conv1.W, conv1_b=self.conv1.b,
            conv2_W=self.conv2.W, conv2_b=self.conv2.b,
            fc1_W=self.fc1.W, fc1_b=self.fc1.b,
            fc2_W=self.fc2.W, fc2_b=self.fc2.b,
            num_classes=self.num_classes
        )

    def load(self, filepath):
        """Load weights from .npz file."""
        data = np.load(filepath)
        self.conv1.W = data['conv1_W']
        self.conv1.b = data['conv1_b']
        self.conv2.W = data['conv2_W']
        self.conv2.b = data['conv2_b']
        self.fc1.W = data['fc1_W']
        self.fc1.b = data['fc1_b']
        self.fc2.W = data['fc2_W']
        self.fc2.b = data['fc2_b']
        self.num_classes = int(data['num_classes'])

class LeNet5Model:
    """Classic Yann LeCun LeNet-5 architecture."""
    def __init__(self, num_classes=36):
        self.num_classes = num_classes
        # C1: Conv 1->6 (5x5, pad=2 so 28x28 -> 28x28)
        self.c1 = Conv2D(1, 6, kernel_size=5, stride=1, padding=2)
        self.r1 = ReLU()
        # S2: Subsampling 2x2 -> 14x14
        self.s2 = MaxPool2D(2, 2)
        # C3: Conv 6->16 (5x5, pad=0 so 14x14 -> 10x10)
        self.c3 = Conv2D(6, 16, kernel_size=5, stride=1, padding=0)
        self.r2 = ReLU()
        # S4: Subsampling 2x2 -> 5x5
        self.s4 = MaxPool2D(2, 2)
        self.flatten = Flatten()
        # C5 / F6: Dense 16*5*5 (400) -> 120
        self.f5 = Dense(400, 120)
        self.r3 = ReLU()
        # F6: Dense 120 -> 84
        self.f6 = Dense(120, 84)
        self.r4 = ReLU()
        # Output: Dense 84 -> num_classes
        self.out = Dense(84, num_classes)

        self.params = [self.c1, self.c3, self.f5, self.f6, self.out]

    def forward(self, x):
        if x.ndim == 3:
            x = x[:, np.newaxis, :, :]
        elif x.ndim == 2:
            x = x[np.newaxis, np.newaxis, :, :]

        out = self.c1.forward(x)
        out = self.r1.forward(out)
        out = self.s2.forward(out)

        out = self.c3.forward(out)
        out = self.r2.forward(out)
        out = self.s4.forward(out)

        out = self.flatten.forward(out)
        out = self.f5.forward(out)
        out = self.r3.forward(out)
        out = self.f6.forward(out)
        out = self.r4.forward(out)
        logits = self.out.forward(out)

        return softmax(logits)

    def save(self, filepath):
        np.savez_compressed(
            filepath,
            c1_W=self.c1.W, c1_b=self.c1.b,
            c3_W=self.c3.W, c3_b=self.c3.b,
            f5_W=self.f5.W, f5_b=self.f5.b,
            f6_W=self.f6.W, f6_b=self.f6.b,
            out_W=self.out.W, out_b=self.out.b,
            num_classes=self.num_classes
        )

    def load(self, filepath):
        data = np.load(filepath)
        self.c1.W = data['c1_W']
        self.c1.b = data['c1_b']
        self.c3.W = data['c3_W']
        self.c3.b = data['c3_b']
        self.f5.W = data['f5_W']
        self.f5.b = data['f5_b']
        self.f6.W = data['f6_W']
        self.f6.b = data['f6_b']
        self.out.W = data['out_W']
        self.out.b = data['out_b']
        self.num_classes = int(data['num_classes'])
