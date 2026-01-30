# PyTorch Backend and Autograd

KlongPy supports multiple array backends. The PyTorch backend enables GPU acceleration and automatic differentiation (autograd) for gradient-based computations.

## Enabling the PyTorch Backend

Set the `USE_TORCH` environment variable:

```bash
# Enable torch backend
USE_TORCH=1 python your_script.py

# Or in the REPL
USE_TORCH=1 kgpy
```

Or programmatically:

```python
import os
os.environ['USE_TORCH'] = '1'

from klongpy import KlongInterpreter
klong = KlongInterpreter()
print(klong._backend.name)  # 'torch'
```

## Backend Comparison

| Feature | NumPy Backend | PyTorch Backend |
|---------|---------------|-----------------|
| Default | Yes | No (requires USE_TORCH=1) |
| Object dtype | Yes | No |
| String operations | Yes | Limited |
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

KlongPy provides two gradient operators:

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

The `∇` operator uses numeric differentiation (finite differences):

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
| `∇` | Numeric | ~1e-6 error | Slower |

With the torch backend (`USE_TORCH=1`), prefer `:>` for:
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

## GPU Acceleration

When CUDA or Apple MPS is available, tensors automatically use GPU:

```python
from klongpy import KlongInterpreter
import os
os.environ['USE_TORCH'] = '1'

klong = KlongInterpreter()
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
import os
os.environ['USE_TORCH'] = '1'

klong = KlongInterpreter()

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

## Troubleshooting

### "PyTorch backend does not support object dtype"

The torch backend cannot handle mixed-type arrays or nested structures with varying shapes. Use the numpy backend for these cases.

### MPS float64 errors

MPS doesn't support float64. The backend automatically converts to float32, but some precision-sensitive operations may behave differently.

### Slow small array operations

For arrays <10K elements, numpy may be faster. Consider using numpy backend for small array workloads or batching small operations together.
