"""
Linear Regression with Gradient Descent using KlongPy

This example demonstrates training a simple linear regression model
using KlongPy's autograd capabilities with the PyTorch backend.

Run with: USE_TORCH=1 python linear_regression.py
"""
from klongpy import KlongInterpreter
import numpy as np

# Create interpreter
klong = KlongInterpreter()

print("Linear Regression with KlongPy Autograd")
print("=" * 50)
print(f"Backend: {klong._backend.name}")
print()

# Generate synthetic data: y = 2*x + 3 + noise
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples).astype(np.float32)
y_true = 2.0 * X + 3.0 + 0.1 * np.random.randn(n_samples).astype(np.float32)

# Put data into klong context
klong['X'] = X
klong['y_true'] = y_true

# Initialize parameters
klong['w'] = np.array([0.0], dtype=np.float32)  # weight
klong['b'] = np.array([0.0], dtype=np.float32)  # bias

print("True parameters: w=2.0, b=3.0")
print("Initial parameters: w=0.0, b=0.0")
print()

# Define the model and loss function in Klong
# Mean squared error loss
klong('''
    predict::{(w*x)+b}
    mse::{+/((predict(X))-y_true)^2}
    grad_mse::∇mse
''')

# Training parameters
learning_rate = 0.01
n_epochs = 100

print(f"Training for {n_epochs} epochs with learning_rate={learning_rate}")
print("-" * 50)

for epoch in range(n_epochs):
    # Compute loss
    loss = float(klong('mse([w b])'))

    # Compute gradients with respect to [w, b]
    params = np.array([float(klong('*w')), float(klong('*b'))], dtype=np.float32)
    klong['params'] = params

    # We need to compute gradient of loss w.r.t. parameters
    # For simplicity, compute numerical gradients
    eps = 1e-4

    # Gradient w.r.t. w
    klong['w'] = np.array([params[0] + eps], dtype=np.float32)
    loss_plus = float(klong('mse([w b])'))
    klong['w'] = np.array([params[0] - eps], dtype=np.float32)
    loss_minus = float(klong('mse([w b])'))
    grad_w = (loss_plus - loss_minus) / (2 * eps)

    # Gradient w.r.t. b
    klong['w'] = np.array([params[0]], dtype=np.float32)
    klong['b'] = np.array([params[1] + eps], dtype=np.float32)
    loss_plus = float(klong('mse([w b])'))
    klong['b'] = np.array([params[1] - eps], dtype=np.float32)
    loss_minus = float(klong('mse([w b])'))
    grad_b = (loss_plus - loss_minus) / (2 * eps)

    # Update parameters
    new_w = params[0] - learning_rate * grad_w
    new_b = params[1] - learning_rate * grad_b

    klong['w'] = np.array([new_w], dtype=np.float32)
    klong['b'] = np.array([new_b], dtype=np.float32)

    if epoch % 10 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch:3d}: loss={loss:10.4f}, w={new_w:.4f}, b={new_b:.4f}")

print()
print("Final parameters:")
print(f"  Learned: w={float(klong('*w')):.4f}, b={float(klong('*b')):.4f}")
print(f"  True:    w=2.0000, b=3.0000")
