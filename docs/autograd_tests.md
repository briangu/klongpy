# Autograd Test Cases

This document explains the mathematical ideas behind the unit tests found in
`tests/test_autograd.py` and their KlongPy counterparts in
`tests/kgtests/autograd`.

KlongPy provides minimal reverse mode automatic differentiation.  The following
examples verify the correctness of the gradient computations for the NumPy and
Torch backends.

## Scalar square

We test the derivative of $f(x)=x^2$.  From the
[definition of the derivative](https://en.wikipedia.org/wiki/Derivative),
$\frac{\mathrm d}{\mathrm dx}x^2=2x$.  The test evaluates this gradient at
$x=3$ and expects the value `6`.

In the Klong test suite the alias ``∂`` is bound to ``backend.grad``.  Calling
``∂(square;3)`` therefore computes the same derivative using the del symbol.

## Matrix multiplication

The function $f(X)=\sum X X$ multiplies a matrix by itself and sums all
elements of the result.  Matrix calculus shows that the derivative of
$\mathrm{tr}(X^2)$ with respect to $X$ is $X+X^T$.
For
$X=\begin{bmatrix}1&2\\3&4\end{bmatrix}$
the gradient is
$\begin{bmatrix}7&11\\9&13\end{bmatrix}$.
See
[the matrix calculus article](https://en.wikipedia.org/wiki/Matrix_calculus)
for details.

## Elementwise product

The function $f(x)=\sum (x+1)(x+2)$ is differentiated using the chain rule
([Wikipedia](https://en.wikipedia.org/wiki/Chain_rule)).  The gradient of each
component is $2x+3$, so the resulting array should equal `2*x + 3`.

## Dot product

For $f(x,y)=\sum x\,y$ (the dot product), the gradient with respect to `x` is
`y` and with respect to `y` is `x`.
See the article on the
[dot product](https://en.wikipedia.org/wiki/Dot_product) for background.

## Stop operator

The `stop` function detaches its argument from the autograd graph.  In
$f(x)=\sum\mathrm{stop}(x)\,x$ the first occurrence of `x` is treated as a
constant, so the gradient simplifies to `x`.
