"""
Linear Regression with Gradient Descent using KlongPy

This example demonstrates training a simple linear regression model
using KlongPy's autograd capabilities with the PyTorch backend.

Run with: python linear_regression.py --backend torch
"""
import argparse

from klongpy import KlongInterpreter
from klongpy.backends import list_backends
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Linear regression with KlongPy autograd and selectable backends."
    )
    parser.add_argument(
        "--backend",
        choices=list_backends(),
        default=None,
        help="Array backend to use (default: numpy).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override (cpu, cuda, mps). Only applies to torch backend.",
    )
    return parser.parse_args()


args = parse_args()

# Create interpreter
klong = KlongInterpreter(backend=args.backend, device=args.device)

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
klong['ytrue'] = y_true
klong['nsamples'] = float(n_samples)

# Initialize parameters
klong['w'] = 0.0  # weight
klong['b'] = 0.0  # bias

print("True parameters: w=2.0, b=3.0")
print("Initial parameters: w=0.0, b=0.0")
print()

# Define the model and loss function in Klong
# Mean squared error loss
klong('''
    predict::{(w*x)+b}
    mse::{(+/((predict(X))-ytrue)^2)%nsamples}
''')

# Training parameters
learning_rate = 0.01
n_epochs = 500

print(f"Training for {n_epochs} epochs with learning_rate={learning_rate}")
print("-" * 50)

for epoch in range(n_epochs):
    # Compute loss
    loss = float(klong('mse(0)'))

    # Compute numerical gradients
    cur_w = float(klong('w'))
    cur_b = float(klong('b'))
    eps = 1e-4

    # Gradient w.r.t. w
    klong['w'] = cur_w + eps
    loss_plus = float(klong('mse(0)'))
    klong['w'] = cur_w - eps
    loss_minus = float(klong('mse(0)'))
    grad_w = (loss_plus - loss_minus) / (2 * eps)

    # Gradient w.r.t. b
    klong['w'] = cur_w
    klong['b'] = cur_b + eps
    loss_plus = float(klong('mse(0)'))
    klong['b'] = cur_b - eps
    loss_minus = float(klong('mse(0)'))
    grad_b = (loss_plus - loss_minus) / (2 * eps)

    # Update parameters
    new_w = cur_w - learning_rate * grad_w
    new_b = cur_b - learning_rate * grad_b

    klong['w'] = new_w
    klong['b'] = new_b

    if epoch % 50 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch:3d}: loss={loss:10.4f}, w={new_w:.4f}, b={new_b:.4f}")

print()
print("Final parameters:")
print(f"  Learned: w={float(klong('w')):.4f}, b={float(klong('b')):.4f}")
print(f"  True:    w=2.0000, b=3.0000")
