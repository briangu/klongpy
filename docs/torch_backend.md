# PyTorch Backend and Autograd

The PyTorch backend runs array operations as torch tensors, on the CPU or a GPU, and gives the gradient operators exact derivatives through autograd. Read this page to decide whether to use it, what changes when you do, and how to use the torch-only functions. NumPy is the default backend and needs no extra install; the torch backend needs `pip install "klongpy[torch]"`.

## Enabling the backend

```bash
kgpy --backend torch                  # best device found: CUDA, MPS or CPU
kgpy --backend torch --device cpu     # a named device
```

From Python:

```python
from klongpy import KlongInterpreter

klong = KlongInterpreter(backend="torch", device="cpu")
print(klong.backend.name)   # torch
```

`device` is `cuda`, `mps` or `cpu`. Without it the backend picks CUDA, then MPS, then the CPU.

## When to use it

| Workload | Backend |
|---|---|
| Gradients, Jacobians, optimization (`:>`, `∂`, `∇`) | torch |
| Arithmetic on arrays of about a million elements or more, on a CPU with many threads | torch |
| Arrays of a few thousand elements, loops, sorting, strings, dictionaries | NumPy |
| Values that need float64 | NumPy |
| GPU | torch; this page has no GPU speed figures, see [Performance](performance.md) for what is measured |

On the CPU, torch pays off only for large arrays and many threads. Sorting and loops dispatch many small operations, which costs more on torch than on NumPy. [Performance](performance.md) has the figures and their conditions; `python tests/perf_backend.py --compare` repeats them on your machine.

## What differs from NumPy

| | NumPy | Torch |
|---|---|---|
| Float width | float64 | float32 on every device, CPU included |
| Gradients (`:>`, `∂`, monad `∇`) | Finite differences | Autograd |
| Strings, mixed lists | numpy object arrays | numpy object arrays on the host, not tensors |
| Results | `numpy.ndarray` | `torch.Tensor` |

Float32 carries about 7 significant decimal digits. `1.5+1e-9` is exactly `1.5` on the torch backend, and sums of many terms lose digits sooner than on NumPy. Autograd has no finite-difference error; float32 rounding still applies.

Known limit: MPS has no float64, and `.gradcheck()` needs the CPU. On MPS it raises `.gradcheck() requires CPU device, got 'mps'`; run `kgpy --backend torch --device cpu` for it.

Known limit: some operations on mixed-type or ragged nested data raise "PyTorch backend does not support object dtype". Simple cases work (`["a" 1]` and `[[1 2][3]]` both evaluate). Use the NumPy backend for the failing operation.

## Gradient operators

`function:>point` is the autograd operator, and so is the monad `∇function`, which returns the gradient as a function. The dyad `point∇function` and `.grad(function;point)` use finite differences on every backend. The table that compares all of them is in the [operator reference](operators.md#comparison); this page shows the torch-only use. The values are those of the torch backend on the CPU.

```klong
f(t)::{t^2}
f:>3                           :" 6.0
g(v)::{+/v^2}
g:>[1.0 2.0 3.0]               :" [2.0 4.0 6.0]
[1 2]∂f                        :" [[2.0 0.0] [0.0 4.0]]
```

A function that reads globals instead of taking an argument is differentiated with respect to them by naming them in a list. The function has no parameters, so what it reads is the value of each name at the time of the call (see [Dynamic scoping](tutorial.md#dynamic-scoping)):

```klong
w::2.0
b::3.0
loss::{(w^2)+(b^2)}
loss:>[w b]                    :" [4.0 6.0]
```

`[w b]∂f` returns one Jacobian per name in the same way. For a function R^n to R^m the Jacobian is an m by n matrix with J[i,j] = df_i/dx_j.

### Typing the gradient symbols

| Symbol | Name | macOS | Linux | Windows |
|---|---|---|---|---|
| `∇` | Nabla (gradient) | Character Viewer: Ctrl+Cmd+Space, search "nabla" | Ctrl+Shift+U, then `2207`, then Enter (GTK and many desktops), or copy-paste | Alt+8711 on the numeric keypad |
| `∂` | Partial (Jacobian) | Option+d | Ctrl+Shift+U, then `2202`, then Enter | Alt+8706 on the numeric keypad |

ASCII alternatives need no special characters. Mind the argument order: `.grad`, `.jacobian` and `:>` take the function first, and the glyphs `∇` and `∂` take the point first.

| Symbol form | ASCII form |
|---|---|
| `3∇f` | `.grad(f;3)` |
| `x∂f` | `.jacobian(f;x)` |
| `f:>x` | (already ASCII) |

### Optimizers

The gradient primitives are `:>`, `∂` and `.jacobian()`. `klongpy.optimizers` provides two optimizers built on them, `SGDOptimizer` (with optional momentum) and `AdamOptimizer`. They are part of the package, so a `pip install` has them.

`loss:>[w]` returns a list with one gradient, so `**` takes its first element: `lr**g` is `lr*(*g)`. (For a single parameter, `loss:>w` returns the gradient itself, so `w::w-(lr*loss:>w)` also works.)

Manual gradient descent needs no optimizer:

```klong
w::10.0
loss::{w^2}
lr::0.1

:" Update rule: w = w - lr * gradient
{w::w-(lr**loss:>[w])}'!50
w                        :" Close to 0 (about 0.0001 after 50 steps)
```

To use an optimizer, import it with `.pyf` from the package. The function receives the interpreter automatically; do not pass `klong` yourself. `SGDOptimizer(names;config)` returns a function that takes the loss function, makes one step and returns the loss before the step. The values are those of the torch backend on the CPU:

<!-- check-docs: requires torch -->
```console
$ cat > sgd.kg <<'EOF'
> .pyf("klongpy.optimizers";"SGDOptimizer")
> w::10.0
> loss::{w^2}
> opt::SGDOptimizer(["w"];:{;"lr";0.1})
> .p(opt(loss))
> {opt(loss)}'!49
> .p(loss())
> EOF
$ kgpy --backend torch --device cpu sgd.kg
100
2.0370366153851167e-08
```

An optimizer works on every device the backend supports: the updated parameters stay on the backend's device (a GPU tensor stays on the GPU).

| Name | Configuration (all optional) |
|---|---|
| `SGDOptimizer` | `"lr"` (0.01), `"momentum"` (0.0) |
| `AdamOptimizer` | `"lr"` (0.001), `"beta1"` (0.9), `"beta2"` (0.999), `"eps"` (1e-8) |

Avoid `x`, `y` and `z` as parameter names: they are the argument names of every function and are shadowed in a call.

An Adam step on two parameters:

```klong
.pyf("klongpy.optimizers";"AdamOptimizer")

w::1.0;b::0.0
X::[1 2 3];Y::[3 5 7]
loss::{(+/((w*X)+b-Y)^2)%3}
opt::AdamOptimizer(["w" "b"];:{;"lr";0.1})

:" Train for 500 steps
{opt(loss)}'!500
```

To write your own update rule (RMSprop, AdaGrad, a learning-rate schedule), copy `klongpy/optimizers.py`: it uses `multi_grad_of_fn` from `klongpy.autograd` to compute the gradients of several parameters at once. `examples/autograd/optimizers.py` re-exports the two optimizers for the example programs, which load them with `.pyf("./optimizers.py";...)`.

## Devices

```python
from klongpy import KlongInterpreter

klong = KlongInterpreter(backend='torch')
print(klong.backend.device)  # 'cuda', 'mps' or 'cpu'
```

Tensors report their device with an index, for example `mps:0`.

## Mixing with Python

`klong(...)` returns a `torch.Tensor` for numeric arrays. Move it to the host before using NumPy:

```python
from klongpy import KlongInterpreter

klong = KlongInterpreter(backend='torch')
result = klong('2*1+!1000000')
print(type(result))   # <class 'torch.Tensor'>
np_result = result.cpu().numpy()
```

See [Python integration](python_integration.md) for the type mapping.

## Practices

- Keep data as tensors; each conversion to NumPy copies.
- Combine operations so that fewer, larger ones run.
- Use `:>` for gradients of functions on the torch backend.

## Function compilation and gradient checks

These functions exist only on the torch backend. They compile a Klong function with `torch.compile`, export its graph and check its gradient. The examples need the torch backend; the values are from torch 2.15.0 on the CPU. Compilation needs a C++ compiler; `"backend" "eager"` skips compilation.

| Function | Does |
|---|---|
| `.compile(fn;input)` | Compiles `fn` for the shape and type of `input` and returns a callable. |
| `.compilex(fn;input;options)` | The same with a dictionary of options: `"mode"`, `"backend"`, `"fullgraph"` (1 to require one graph) and `"dynamic"` (1 for dynamic shapes, 0 for static). |
| `.cmodes()` | A dictionary with the keys `modes`, `backends` and `recommendations`. |
| `.export(fn;input;path)` | Exports the computation graph to `path` and returns a dictionary with `compiled_fn`, `export_path`, `graph`, `graph_module`, `mode` and `backend`. |
| `.gradcheck(fn;input)` | Returns 1 when the autograd gradient of `fn` at `input` matches the numeric one. CPU only. |

```klong
f(t)::{t^2}
cf::.compile(f;3.0)
cf(5.0)                                      :" 25.0
ce::.compilex(f;3.0;:{["backend" "eager"]})  :" no compilation
.cmodes()@"modes"                            :" the modes, with a description of each
.gradcheck(f;3.0)                            :" 1
```

The modes are `default`, `reduce-overhead` (faster to compile, less optimisation) and `max-autotune` (slowest to compile, most optimisation). The backends are `inductor` (the default, C++ or Triton code), `eager` (no compilation), `aot_eager` (eager with the autograd graph) and `cudagraphs` (GPU only).

`.export` writes a file that `torch.export.load(path).module()` reads back. If torch cannot export the function, `export_path` is empty, `export_error` holds the reason and `compiled_fn` is still usable.

Known limit: give each compiled function its own name; assigning a second compiled function to the same name is not supported.

## Troubleshooting

- `.compile()` fails with C++ errors: install a C++ compiler (clang++ or g++) and, on macOS, the Xcode Command Line Tools. `"backend" "eager"` skips compilation.
- Results differ from NumPy in the last digits: float32 (see above).

## See also

- [Roadmap](roadmap.md) for the known limits of the torch backend and what is planned.
- [Performance](performance.md) and [Python integration](python_integration.md).
- [Migrating to v2](migrating-to-v2.md) and [What's new in v2](whats-new-v2.md).
