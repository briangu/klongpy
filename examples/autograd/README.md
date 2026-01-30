# KlongPy Autograd Examples

These examples demonstrate KlongPy's automatic differentiation capabilities using the PyTorch backend.

## Prerequisites

Install KlongPy with PyTorch support:

```bash
pip install klongpy torch
```

## Running Examples

All autograd examples require the PyTorch backend. Set the `USE_TORCH=1` environment variable:

```bash
# Python examples
USE_TORCH=1 python basic_gradient.py
USE_TORCH=1 python linear_regression.py

# Klong language examples
USE_TORCH=1 kgpy basic_gradient.kg
USE_TORCH=1 kgpy linear_regression.kg
```

## Examples

### basic_gradient.py / basic_gradient.kg

Demonstrates fundamental gradient computation:
- Computing gradients of simple functions (x^2, polynomials)
- Gradients with array inputs
- Chain rule with composed functions

### linear_regression.py / linear_regression.kg

Shows a practical machine learning example:
- Training a linear regression model
- Using gradients for parameter optimization
- Comparing learned vs true parameters

## Using Autograd in KlongPy

The autograd operator `:>` computes the gradient of a function at a point:

```klong
f::{x^2}        :" Define a function
f:>3            :" Compute gradient at x=3 -> 6.0
```

The syntax is `function:>point` where:
- `function` is a scalar-valued function (must return a single number)
- `point` is the input at which to compute the gradient

For functions with array inputs, the gradient is computed element-wise:

```klong
g::{+/x^2}           :" Sum of squares: g(x) = x1^2 + x2^2 + ...
g:>[1.0 2.0 3.0]     :" Returns [2 4 6] = [2*x1, 2*x2, 2*x3]
```

## Examples

### Scalar function gradient
```klong
f::{x^3}         :" f(x) = x^3
f:>2             :" f'(2) = 3*4 = 12.0
```

### Polynomial gradient
```klong
p::{((3*x^4)-(2*x^2))+x}   :" p(x) = 3x^4 - 2x^2 + x
p:>1                        :" p'(1) = 12 - 4 + 1 = 9.0
```

### Gradient descent
```klong
f::{x^2}
w::5.0
lr::0.1
:" Update step: w = w - lr * gradient
w::w-(lr*f:>w)
```

## Performance

When using the torch backend with `USE_TORCH=1`:
- Automatic differentiation is computed using PyTorch's autograd
- GPU acceleration is available when CUDA or MPS is present
- Large array operations benefit from torch's optimized kernels

Without torch (numpy backend):
- The `:>` operator falls back to numeric differentiation
- Uses finite differences: f'(x) ≈ (f(x+ε) - f(x-ε)) / 2ε
- Still functional but slower and less precise for complex functions
