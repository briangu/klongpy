# KlongPy Development Guidelines

## Torch isolation

**Do not import torch outside of `klongpy/backends/torch_backend.py`.**

All torch-specific code must be confined to the torch backend module. Other modules (interpreter, compiler, monads, dyads, adverbs) must not `import torch` directly. When torch functionality is needed elsewhere:

- Use `klong._backend.np.ndarray` to check for tensors (resolves to `torch.Tensor` on torch backend, `numpy.ndarray` on numpy)
- Use `hasattr(klong._backend, '_torch_backend')` to detect the torch backend
- Use `sys.modules.get('torch')` to get the torch module reference when the torch backend is already active
- Use backend methods (e.g., `klong._backend.np.cumsum`) for operations that differ between backends

This keeps torch as an optional dependency and prevents import-time failures when torch is not installed.

## Testing

Run the full test suite with:
```bash
pip install -e ".[full]"
python -m unittest discover tests
```
