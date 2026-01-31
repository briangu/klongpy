"""
Example optimizer classes for KlongPy.

Copy this file to your project and customize as needed.

Usage in Klong:
    .pyf("optimizers";"SGDOptimizer")
    opt::SGDOptimizer(klong;[w b];:{["lr" 0.1]})
    {opt(loss)}'!100

Or from Python:
    from optimizers import SGDOptimizer
    opt = SGDOptimizer(klong, ['w', 'b'], {'lr': 0.1})
    for _ in range(100):
        opt(klong['loss'])
"""
import numpy as np
from klongpy.core import KGSym, KGFn, KGCall


class SGDOptimizer:
    """
    Stochastic Gradient Descent with optional momentum.

    Args:
        klong: KlongInterpreter instance
        params: List of parameter symbols ['w', 'b'] or [KGSym('w'), KGSym('b')]
        config: Dict with 'lr' (learning rate, default 0.01) and
                'momentum' (default 0.0)

    Example:
        opt = SGDOptimizer(klong, ['w', 'b'], {'lr': 0.1, 'momentum': 0.9})
        opt(loss_fn)  # Computes gradients and updates parameters
    """

    def __init__(self, klong, params, config=None):
        self.klong = klong
        # Convert string params to KGSym
        self.params = [KGSym(p) if isinstance(p, str) else p for p in params]
        config = config if config is not None else {}
        self.lr = config.get('lr', 0.01)
        self.momentum = config.get('momentum', 0.0)
        self.velocities = {p: 0.0 for p in self.params}

    def __call__(self, loss_fn):
        """
        Perform one optimization step.

        Args:
            loss_fn: Loss function (KGSym, KGFn, or callable)

        Returns:
            Loss value before the update
        """
        from klongpy.autograd import multi_grad_of_fn

        # Compute loss before update (for return value)
        if isinstance(loss_fn, KGFn):
            loss = self.klong.call(loss_fn)
        elif isinstance(loss_fn, KGSym):
            loss = self.klong.call(KGCall(loss_fn, [], 0))
        else:
            loss = loss_fn()

        # Compute gradients for all parameters
        grads = multi_grad_of_fn(self.klong, loss_fn, self.params)

        # Update parameters with momentum
        backend = self.klong._backend
        for param, grad in zip(self.params, grads):
            # Convert grad to numpy if needed
            if backend.is_backend_array(grad):
                grad = backend.to_numpy(grad)
            grad = np.asarray(grad)

            # Apply momentum: v = momentum * v + grad
            v = self.momentum * self.velocities[param] + grad
            self.velocities[param] = v

            # Update parameter: param = param - lr * v
            current = self.klong[param]
            if backend.is_backend_array(current):
                current = backend.to_numpy(current)
            new_val = current - self.lr * v
            self.klong[param] = float(new_val) if np.ndim(new_val) == 0 else new_val

        return loss

    def __str__(self):
        return f"SGDOptimizer(lr={self.lr}, momentum={self.momentum})"

    def __repr__(self):
        return str(self)


class AdamOptimizer:
    """
    Adam optimizer with adaptive learning rates.

    Args:
        klong: KlongInterpreter instance
        params: List of parameter symbols ['w', 'b'] or [KGSym('w'), KGSym('b')]
        config: Dict with:
            - 'lr': learning rate (default 0.001)
            - 'beta1': first moment decay (default 0.9)
            - 'beta2': second moment decay (default 0.999)
            - 'eps': numerical stability (default 1e-8)

    Example:
        opt = AdamOptimizer(klong, ['w', 'b'], {'lr': 0.001})
        opt(loss_fn)  # Computes gradients and updates parameters
    """

    def __init__(self, klong, params, config=None):
        self.klong = klong
        self.params = [KGSym(p) if isinstance(p, str) else p for p in params]
        config = config if config is not None else {}
        self.lr = config.get('lr', 0.001)
        self.beta1 = config.get('beta1', 0.9)
        self.beta2 = config.get('beta2', 0.999)
        self.eps = config.get('eps', 1e-8)
        self.t = 0  # Time step
        self.m = {p: 0.0 for p in self.params}  # First moment estimates
        self.v = {p: 0.0 for p in self.params}  # Second moment estimates

    def __call__(self, loss_fn):
        """
        Perform one optimization step.

        Args:
            loss_fn: Loss function (KGSym, KGFn, or callable)

        Returns:
            Loss value before the update
        """
        from klongpy.autograd import multi_grad_of_fn

        self.t += 1

        # Compute loss before update
        if isinstance(loss_fn, KGFn):
            loss = self.klong.call(loss_fn)
        elif isinstance(loss_fn, KGSym):
            loss = self.klong.call(KGCall(loss_fn, [], 0))
        else:
            loss = loss_fn()

        # Compute gradients
        grads = multi_grad_of_fn(self.klong, loss_fn, self.params)

        # Update parameters
        backend = self.klong._backend
        for param, grad in zip(self.params, grads):
            if backend.is_backend_array(grad):
                grad = backend.to_numpy(grad)
            g = np.asarray(grad)

            # Update biased first moment estimate
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * g
            # Update biased second raw moment estimate
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (g ** 2)

            # Compute bias-corrected estimates
            m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)

            # Update parameter
            current = self.klong[param]
            if backend.is_backend_array(current):
                current = backend.to_numpy(current)
            new_val = current - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            self.klong[param] = float(new_val) if np.ndim(new_val) == 0 else new_val

        return loss

    def __str__(self):
        return f"AdamOptimizer(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2})"

    def __repr__(self):
        return str(self)
