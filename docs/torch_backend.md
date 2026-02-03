# PyTorch Backend and Autograd

KlongPy supports multiple array backends. The PyTorch backend enables GPU acceleration and automatic differentiation (autograd) for gradient-based computations.

## Enabling the PyTorch Backend

### Command Line

```bash
# Use --backend flag
kgpy --backend torch

# With GPU device selection
kgpy --backend torch --device cuda
```

### Programmatically

```python
from klongpy import KlongInterpreter

# Create interpreter with torch backend
klong = KlongInterpreter(backend="torch")
print(klong._backend.name)  # 'torch'

# With specific device
klong = KlongInterpreter(backend="torch", device="cuda")
```

## Backend Comparison

| Feature | NumPy Backend | PyTorch Backend |
|---------|---------------|-----------------|
| Default | Yes | No (use `--backend torch`) |
| Object dtype | Yes | No |
| String operations | Yes | Not supported |
| GPU acceleration | No | Yes (CUDA/MPS) |
| Autograd | Numeric only | Native autograd |
| Small array performance | Faster | Slightly slower |
| Large array performance | Good | Better (especially on GPU) |

## Performance

The torch backend excels with large arrays:

```
Benchmark              NumPy      Torch      Winner
---------------------------------------------------------
vector_add_100K        0.04ms     0.08ms     NumPy (2x)
vector_add_1M          0.36ms     0.07ms     Torch (5x)
compound_expr_1M       0.61ms     0.07ms     Torch (8x)
grade_up_100K          0.59ms     0.19ms     Torch (3x)
```

For small arrays (<100K elements), NumPy is slightly faster due to lower dispatch overhead. For larger arrays, torch wins significantly.

## Automatic Differentiation

KlongPy provides several gradient and differentiation operators:

### Typing Special Characters

| Symbol | Name | Mac | Windows |
|--------|------|-----|---------|
| `∇` | Nabla | Character Viewer (Ctrl+Cmd+Space) | Alt+8711 |
| `∂` | Partial | **Option + d** | Alt+8706 |

On Mac, `∂` can be typed directly with **Option + d**. For `∇`, use the Character Viewer or copy-paste.

### `:>` Autograd Operator (Recommended)

The `:>` operator uses PyTorch autograd for exact gradients:

```klong
f::{x^2}         :" Define f(x) = x^2
f:>3             :" Compute f'(3) = 6.0
```

The syntax is `function:>point` where:
- `function` is a scalar-valued function (must return a single number)
- `point` is the input at which to compute the gradient

### `∇` Numeric Gradient Operator

The `∇` operator **always** uses numeric differentiation (finite differences), regardless of backend:

```klong
f::{x^2}         :" Define f(x) = x^2
3∇f              :" Compute f'(3) ≈ 6.0
```

The syntax is `point∇function` (note: reversed order from `:>`).

### How They Work

| Operator | Method | Precision | Speed |
|----------|--------|-----------|-------|
| `:>` with torch | PyTorch autograd | Exact | Fast |
| `:>` without torch | Numeric | ~1e-6 error | Slower |
| `∇` (any backend) | Always numeric | ~1e-6 error | Slower |

With the torch backend (`--backend torch` or `backend='torch'`), prefer `:>` for:
- Exact gradients (no floating-point approximation error)
- Complex computational graphs
- Better performance on large arrays

### Examples

**Scalar function:**
```klong
f::{x^3}          :" f(x) = x^3
f:>2              :" f'(2) = 3*4 = 12.0
```

**Polynomial:**
```klong
p::{((3*x^4)-(2*x^2))+x}   :" p(x) = 3x^4 - 2x^2 + x
p:>1                        :" p'(1) = 12 - 4 + 1 = 9.0
```

**Vector function (sum of squares):**
```klong
g::{+/x^2}             :" g(x) = sum(x_i^2)
g:>[1.0 2.0 3.0]       :" [2 4 6] = 2*x
```

**Gradient descent:**
```klong
f::{x^2}
x::5.0
lr::0.1

:" Update rule: x = x - lr * grad
x::x-(lr*f:>x)
```

### Multi-Parameter Gradients

Compute gradients for multiple parameters simultaneously using a list of symbols:

```klong
w::2.0
b::3.0
loss::{(w^2)+(b^2)}

:" Compute gradients for both w and b
grads::loss:>[w b]    :" [4.0 6.0] = [2w, 2b]
```

This is especially useful for neural network training:

```klong
w::1.0
b::0.0
X::[1 2 3]
Y::[3 5 7]

:" MSE loss
loss::{(+/((w*X)+b-Y)^2)%3}

:" Compute both gradients in one call
grads::loss:>[w b]
```

### Jacobian Computation

Compute the Jacobian matrix (matrix of partial derivatives) using the `∂` operator or `.jacobian()` function:

```klong
f::{x^2}                 :" Element-wise square

:" Using ∂ operator (point∂function)
[1 2]∂f                  :" [[2 0] [0 4]] diagonal matrix

:" Using .jacobian() function
.jacobian(f;[1 2])       :" Same result
```

For vector-valued functions f: R^n -> R^m, the Jacobian is an m x n matrix where J[i,j] = df_i/dx_j.

### Multi-Parameter Jacobians

Just like gradients, you can compute Jacobians with respect to multiple parameters using a list of symbols:

```klong
w::[1.0 2.0]
b::[3.0 4.0]
f::{w^2}                 :" Returns [w0^2, w1^2]

:" Compute Jacobians for both w and b
jacobians::[w b]∂f       :" Returns [J_w, J_b]
```

This returns a list of Jacobian matrices, one per parameter. Useful for analyzing how vector-valued functions depend on multiple parameter sets.

### Custom Optimizers

KlongPy provides the gradient primitives (`:>`, `∂`, `.jacobian()`). For optimizers, use the example classes in `examples/autograd/optimizers.py` which you can copy to your project and customize.

**Manual gradient descent (no optimizer needed):**
```klong
w::10.0
loss::{w^2}
lr::0.1

:" Update rule: w = w - lr * gradient
{w::w-(lr*loss:>w)}'!50
w                        :" Close to 0
```

**Using a custom optimizer class:**

1. Copy `examples/autograd/optimizers.py` to your project directory
2. Import with `.pyf()`:

```klong
:" Import the optimizer class
.pyf("optimizers";"SGDOptimizer")

:" Setup parameters and loss
w::10.0
loss::{w^2}

:" Create optimizer with learning rate 0.1
opt::SGDOptimizer(klong;["w"];:{["lr" 0.1]})

:" Run optimization steps
{opt(loss)}'!50
w                        :" Close to 0
```

**Available example optimizers:**
- `SGDOptimizer` - Stochastic Gradient Descent with optional momentum
- `AdamOptimizer` - Adam optimizer with adaptive learning rates

**SGD with momentum:**
```klong
.pyf("optimizers";"SGDOptimizer")
opt::SGDOptimizer(klong;["w"];:{["lr" 0.01 "momentum" 0.9]})
```

**Adam optimizer:**
```klong
.pyf("optimizers";"AdamOptimizer")
opt::AdamOptimizer(klong;["w" "b"];:{["lr" 0.001]})
```

**Training loop example:**
```klong
.pyf("optimizers";"AdamOptimizer")

w::1.0;b::0.0
X::[1 2 3];Y::[3 5 7]
loss::{(+/((w*X)+b-Y)^2)%3}
opt::AdamOptimizer(klong;["w" "b"];:{["lr" 0.1]})

:" Train for 500 steps
{opt(loss)}'!500
```

**Creating your own optimizer:**

The example optimizers use `multi_grad_of_fn` from `klongpy.autograd` to compute gradients for multiple parameters. Copy and modify the optimizer classes to implement custom update rules (RMSprop, AdaGrad, learning rate schedules, etc.).

## GPU Acceleration

When CUDA or Apple MPS is available, tensors automatically use GPU:

```python
from klongpy import KlongInterpreter

klong = KlongInterpreter(backend='torch')
print(klong._backend.device)  # 'cuda:0', 'mps:0', or 'cpu'
```

### Device Selection

The backend automatically selects the best available device:
1. CUDA (NVIDIA GPU) - if available
2. MPS (Apple Silicon) - if available
3. CPU - fallback

### MPS Limitations

Apple's MPS backend has some limitations:
- No float64 support (uses float32)
- Some operations fall back to CPU

## Mixing with Python

Access torch tensors directly:

```python
from klongpy import KlongInterpreter

klong = KlongInterpreter(backend='torch')

# KlongPy operations return torch tensors
result = klong('2*1+!1000000')
print(type(result))  # <class 'torch.Tensor'>
print(result.device)  # cuda:0, mps:0, or cpu

# Convert to numpy when needed
import numpy as np
np_result = result.cpu().numpy()
```

## Best Practices

1. **Use torch for large computations**: Switch to torch backend for arrays >100K elements

2. **Keep data as tensors**: Avoid unnecessary conversions between numpy and torch

3. **Batch operations**: Combine operations to minimize dispatch overhead

4. **Use autograd for gradients**: Native autograd is faster and more accurate than numeric differentiation

## Function Compilation

The torch backend supports compiling Klong functions for optimized execution using `torch.compile`:

### `.compile(fn;input)` - Compile Function

Compiles a function for faster execution:

```klong
f::{x^2}
cf::.compile(f;3.0)      :" Returns compiled function
cf(5.0)                   :" 25.0 (optimized)
```

The compiled function runs significantly faster for complex computations.

### `.export(fn;input;path)` - Export Computation Graph

Exports the function's computation graph to a file for inspection:

```klong
f::{(x^3)+(2*x^2)+x}
info::.export(f;2.0;"model.pt2")
.p(info@"graph")         :" Print computation graph
```

Returns a dictionary with:
- `"compiled_fn"` - The compiled function
- `"export_path"` - Path where graph was saved
- `"graph"` - String representation of computation graph

The exported `.pt2` file can be loaded with `torch.export.load()` in Python.

### `.compilex(fn;input;options)` - Extended Compilation

Compile with advanced options for mode and backend:

```klong
f::{x^2}

:" Fast compilation for development
cf::.compilex(f;3.0;:{["mode" "reduce-overhead"]})

:" Maximum optimization for production
cf::.compilex(f;3.0;:{["mode" "max-autotune"]})

:" Debug mode (no compilation)
cf::.compilex(f;3.0;:{["backend" "eager"]})
```

**Options dictionary:**
- `"mode"` - Compilation mode (see table below)
- `"backend"` - Compilation backend (see table below)
- `"fullgraph"` - Set to 1 to require full graph compilation
- `"dynamic"` - Set to 1 for dynamic shapes, 0 for static

### `.cmodes()` - Query Compilation Modes

Get information about available modes and backends:

```klong
info::.cmodes()
.p(info@"modes")          :" Available compilation modes
.p(info@"backends")       :" Available backends
.p(info@"recommendations") :" Suggested settings
```

### Compilation Mode Comparison

| Mode              | Compile Time | Runtime Speed | Best For           |
|-------------------|--------------|---------------|---------------------|
| `default`         | Medium       | Good          | General use         |
| `reduce-overhead` | Fast         | Moderate      | Development/testing |
| `max-autotune`    | Slow         | Best          | Production          |

### Backend Comparison

| Backend      | Description                                      |
|--------------|--------------------------------------------------|
| `inductor`   | Default - C++/Triton code generation (fastest)   |
| `eager`      | No compilation - runs original Python (debugging)|
| `aot_eager`  | Ahead-of-time eager (debugging + autograd)       |
| `cudagraphs` | CUDA graphs - reduces GPU kernel launch overhead |

**Note:** Compilation requires a C++ compiler on your system. Use `"backend" "eager"` to bypass compilation for debugging. If compilation fails, an error message will indicate the issue.

## Gradient Verification

Use `.gradcheck()` to verify that autograd gradients are correct:

### `.gradcheck(fn;inputs)` - Verify Gradients

Verifies autograd gradients against numeric gradients:

```klong
f::{x^2}
.gradcheck(f;3.0)        :" Returns 1 if correct

g::{+/x^2}
.gradcheck(g;[1.0 2.0 3.0])  :" Returns 1
```

This uses `torch.autograd.gradcheck` internally for rigorous verification.

**Use cases:**
- Verifying custom gradient implementations
- Debugging gradient computation issues
- Ensuring numerical stability

## Troubleshooting

### "PyTorch backend does not support object dtype"

The torch backend cannot handle mixed-type arrays or nested structures with varying shapes. Use the numpy backend for these cases.

### MPS float64 errors

MPS doesn't support float64. The backend automatically converts to float32, but some precision-sensitive operations may behave differently.

### Slow small array operations

For arrays <10K elements, numpy may be faster. Consider using numpy backend for small array workloads or batching small operations together.

### torch.compile errors

If `.compile()` fails with C++ errors, ensure you have:
- A C++ compiler installed (clang++ or g++)
- The required header files (may need Xcode Command Line Tools on macOS)
