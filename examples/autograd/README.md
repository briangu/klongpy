# KlongPy Autograd Examples

These examples demonstrate KlongPy's automatic differentiation capabilities.

## Prerequisites

Install KlongPy:

```bash
pip install klongpy
```

For PyTorch autograd support (optional, for exact gradients):

```bash
pip install klongpy torch
```

## Running Examples

All examples work with any backend. The `:>` operator falls back to numeric differentiation when torch is not available:

```bash
# Klong language examples (work without torch)
kgpy basic_gradient.kg
kgpy numeric_vs_autograd.kg
kgpy gradient_descent.kg
kgpy linear_regression.kg
kgpy portfolio_opt.kg
kgpy neural_net.kg
kgpy optimizer_demo.kg

# With PyTorch backend for exact gradients (recommended)
USE_TORCH=1 kgpy basic_gradient.kg
USE_TORCH=1 kgpy optimizer_demo.kg

# Python examples (require torch for autograd)
USE_TORCH=1 python basic_gradient.py
USE_TORCH=1 python linear_regression.py
```

## Klong Examples

### numeric_vs_autograd.kg
Comparison of the two gradient operators:
- `∇` (nabla): Always numeric differentiation (syntax: `point∇function`)
- `:>`: PyTorch autograd when available, numeric fallback (syntax: `function:>point`)

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

### optimizer_demo.kg
Using custom optimizer classes:
- Simple quadratic minimization with SGD
- Linear regression with Adam optimizer
- Manual gradient descent with multi-param gradients

### optimizers.py
Reusable optimizer classes for your projects:
- `SGDOptimizer` - SGD with optional momentum
- `AdamOptimizer` - Adam with adaptive learning rates
- Copy this file to your project and customize as needed

## Python Examples

### basic_gradient.py
Python version of gradient computations with interpreter integration.

### linear_regression.py
Python version with NumPy integration for data generation.

## Using Autograd in KlongPy

Two gradient operators are available:

### `∇` (nabla) - Always Numeric Differentiation

The `∇` operator **always** uses finite differences, regardless of backend:

```klong
f::{x^2}        :" Define a function
3∇f             :" Compute gradient at x=3 -> ~6.0
```

The syntax is `point∇function`.

### `:>` - Autograd

```klong
f::{x^2}        :" Define a function
f:>3            :" Compute gradient at x=3 -> 6.0
```

The syntax is `function:>point`. Uses PyTorch autograd when `USE_TORCH=1`, otherwise falls back to numeric differentiation.

### Gradient Descent Pattern

```klong
:" Define loss function
loss::{(x-target)^2}

:" Initialize parameter
theta::10.0
lr::0.1

:" Training step
step::{grad::loss:>theta;theta::theta-(lr*grad)}

:" Train for N epochs
step'!100
```

### Multi-Parameter Gradients

Use the `:>[w b]` syntax to compute gradients for multiple parameters at once:

```klong
:" Parameters
w::2.0
b::3.0

:" Loss function using both parameters
loss::{(w^2)+(b^2)}

:" Compute gradients for both in one call
grads::loss:>[w b]    :" Returns [4.0 6.0] = [2w, 2b]

:" Update parameters
lr::0.1
w::w-(lr*grads@0)
b::b-(lr*grads@1)
```

### Custom Optimizers

For more sophisticated optimization, use the optimizer classes provided in `optimizers.py`:

```klong
:" Import optimizer class
.pyf("optimizers";"SGDOptimizer")

:" Setup
w::10.0
loss::{w^2}

:" Create and use optimizer
opt::SGDOptimizer(klong;["w"];:{["lr" 0.1]})
{opt(loss)}'!50       :" Run 50 optimization steps
```

Available optimizer classes (copy and customize as needed):
- `SGDOptimizer` - Stochastic Gradient Descent with optional momentum
- `AdamOptimizer` - Adam optimizer with adaptive learning rates

See `optimizer_demo.kg` for complete examples.

## Performance

When using the torch backend with `USE_TORCH=1`:
- Automatic differentiation uses PyTorch's autograd
- GPU acceleration available (CUDA/MPS)
- Large array operations benefit from torch's optimized kernels

Without torch (numpy backend):
- Both `∇` and `:>` use numeric differentiation
- Uses finite differences: f'(x) ≈ (f(x+ε) - f(x-ε)) / 2ε
- Functional but slower and less precise than exact gradients

## Tips

1. **Start simple**: Test gradients on simple functions first
2. **Check dimensions**: Ensure loss function returns a scalar
3. **Learning rate**: Start with small values (0.001-0.1)
4. **Constraints**: Use penalty terms or project after each step
5. **Debugging**: Print loss every N steps to verify convergence
