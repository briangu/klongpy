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
# Klong language examples
USE_TORCH=1 kgpy basic_gradient.kg
USE_TORCH=1 kgpy gradient_descent.kg
USE_TORCH=1 kgpy linear_regression.kg
USE_TORCH=1 kgpy portfolio_opt.kg
USE_TORCH=1 kgpy neural_net.kg

# Python examples
USE_TORCH=1 python basic_gradient.py
USE_TORCH=1 python linear_regression.py
```

## Klong Examples

### numeric_vs_autograd.kg
Comparison of the two gradient operators:
- `∇` (nabla): Numeric differentiation
- `:>`: PyTorch autograd
- Precision and syntax differences
- Works with any backend (no torch required)

### basic_gradient.kg
Fundamental gradient computation:
- Gradients of simple functions (x^2, polynomials)
- Gradients with array inputs
- Chain rule with composed functions

### gradient_descent.kg
Core gradient descent concepts:
- Minimizing simple functions
- Rosenbrock function optimization
- Linear regression fitting
- Quadratic curve fitting

### linear_regression.kg
Complete machine learning example:
- Training a linear regression model
- Gradient descent optimization
- Comparing learned vs true parameters

### portfolio_opt.kg
Finance application:
- Mean-variance portfolio optimization
- Sharpe ratio maximization
- Constraint handling with penalties
- Multi-asset allocation

### neural_net.kg
Neural network fundamentals:
- Single neuron (perceptron) learning AND gate
- Function approximation with hidden layer
- Learning sin(x) with a 2-layer network

## Python Examples

### basic_gradient.py
Python version of gradient computations with interpreter integration.

### linear_regression.py
Python version with NumPy integration for data generation.

## Using Autograd in KlongPy

The autograd operator `:>` computes the gradient of a function at a point:

```klong
f::{x^2}        :" Define a function
f:>3            :" Compute gradient at x=3 -> 6.0
```

The syntax is `function:>point` where:
- `function` is a scalar-valued function (must return a single number)
- `point` is the input at which to compute the gradient

### Gradient Descent Pattern

```klong
:" Define loss function
loss::{(x-target)^2}

:" Initialize parameter
theta::10.0
lr::0.1

:" Training step
step::{
    grad::loss:>theta
    theta::theta-(lr*grad)
}

:" Train for N epochs
step'!100
```

### Multi-Parameter Optimization

For functions with multiple parameters, define separate loss functions:

```klong
:" Parameters
w::0.0
b::0.0

:" Loss as function of each parameter (holding others constant)
lossW::{mse(x;b)}
lossB::{mse(w;x)}

:" Update each parameter
step::{
    gw::lossW:>w
    gb::lossB:>b
    w::w-(lr*gw)
    b::b-(lr*gb)
}
```

## Performance

When using the torch backend with `USE_TORCH=1`:
- Automatic differentiation uses PyTorch's autograd
- GPU acceleration available (CUDA/MPS)
- Large array operations benefit from torch's optimized kernels

Without torch (numpy backend):
- Falls back to numeric differentiation
- Uses finite differences: f'(x) ≈ (f(x+ε) - f(x-ε)) / 2ε
- Functional but slower and less precise

## Tips

1. **Start simple**: Test gradients on simple functions first
2. **Check dimensions**: Ensure loss function returns a scalar
3. **Learning rate**: Start with small values (0.001-0.1)
4. **Constraints**: Use penalty terms or project after each step
5. **Debugging**: Print loss every N steps to verify convergence

## Advanced Examples

For more advanced examples including trading strategies with autograd, see the `klongpy-trading-autograd.zip` archive which includes:
- Differentiable parameter optimization
- Learned signal combination
- Statistical arbitrage with learned hedge ratios
- Portfolio optimization with constraints
- Neural network price prediction
