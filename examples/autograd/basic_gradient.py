"""
Basic Gradient Computation with KlongPy

This example demonstrates computing gradients of simple functions
using KlongPy's autograd capabilities with the PyTorch backend.

Run with: python basic_gradient.py --backend torch
"""
import argparse

from klongpy import KlongInterpreter
from klongpy.backends import list_backends
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="KlongPy autograd example using configurable backends."
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

# Create interpreter (use --backend torch to enable PyTorch)
klong = KlongInterpreter(backend=args.backend, device=args.device)

print("KlongPy Autograd Examples")
print("=" * 50)
print(f"Backend: {klong._backend.name}")
print(f"Autograd supported: {klong._backend.supports_autograd()}")
print()

# Example 1: Gradient of x^2 at x=3
# The derivative of x^2 is 2x, so at x=3 the gradient should be 6
print("Example 1: Gradient of x^2 at x=3")
print("-" * 40)

klong('f::{x^2}')  # Define f(x) = x^2
klong('grad_f::∇f')  # Get the gradient function

result = klong('grad_f(3)')
print(f"f(x) = x^2")
print(f"f'(x) = 2x")
print(f"f'(3) = {result}")
print(f"Expected: 6.0")
print()

# Example 2: Gradient of a more complex function
# f(x) = x^3 + 2*x^2 - 5*x
# f'(x) = 3*x^2 + 4*x - 5
print("Example 2: Gradient of x^3 + 2*x^2 - 5*x at x=2")
print("-" * 40)

klong('g::{(x^3)+(2*x^2)-(5*x)}')
klong('grad_g::∇g')

result = klong('grad_g(2)')
print(f"g(x) = x^3 + 2x^2 - 5x")
print(f"g'(x) = 3x^2 + 4x - 5")
print(f"g'(2) = {result}")
print(f"Expected: 3*4 + 4*2 - 5 = 15.0")
print()

# Example 3: Gradient with array input
print("Example 3: Gradient of sum of squares")
print("-" * 40)

klong('h::{+/x^2}')  # h(x) = sum(x^2)
klong('grad_h::∇h')

# For h(x) = x1^2 + x2^2 + x3^2, gradient is [2*x1, 2*x2, 2*x3]
x = np.array([1.0, 2.0, 3.0])
klong['x'] = x
result = klong('grad_h(x)')

print(f"h(x) = sum(x^2) = x1^2 + x2^2 + x3^2")
print(f"∇h(x) = [2*x1, 2*x2, 2*x3]")
print(f"x = {x}")
print(f"∇h(x) = {result}")
print(f"Expected: [2.0, 4.0, 6.0]")
print()

# Example 4: Using gradient in optimization (simple gradient descent)
print("Example 4: Simple Gradient Descent to minimize x^2")
print("-" * 40)

klong('f::{x^2}')
klong('grad_f::∇f')

# Gradient descent: x_new = x - learning_rate * gradient
x = 5.0
learning_rate = 0.1

print(f"Starting x = {x}")
print(f"Learning rate = {learning_rate}")
print()

for i in range(10):
    klong['x'] = x
    grad = float(klong('grad_f(x)'))
    x = x - learning_rate * grad
    loss = float(klong('f(x)'))
    print(f"Step {i+1}: x = {x:.6f}, f(x) = {loss:.6f}, grad = {grad:.6f}")

print()
print(f"Final x = {x:.6f} (should approach 0)")
