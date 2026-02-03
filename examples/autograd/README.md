# KlongPy Autograd Examples

These examples demonstrate KlongPy's automatic differentiation capabilities, from basic gradients to real-world applications in finance and physics.

## Prerequisites

Install KlongPy with PyTorch backend:

```bash
pip install klongpy torch
```

For examples using live market data:

```bash
pip install yfinance
```

## Quick Start

All examples require the torch backend for autograd:

```bash
kgpy --backend torch <example>.kg
```

---

## Examples Overview

| Example | Description | Prerequisites |
|---------|-------------|---------------|
| `basic_gradient.kg` | Fundamental gradient computation | torch |
| `numeric_vs_autograd.kg` | Compare numeric vs autograd | torch |
| `gradient_descent.kg` | Optimization algorithms | torch |
| `linear_regression.kg` | ML training example | torch |
| `neural_net.kg` | Neural network from scratch | torch |
| `portfolio_opt.kg` | Sharpe ratio optimization | torch, yfinance |
| `optimizer_demo.kg` | Custom SGD/Adam optimizers | torch |
| `black_scholes_greeks.kg` | Option Greeks via autograd | torch |
| `black_scholes_live.kg` | Live market verification | torch, yfinance |
| `differentiable_physics.kg` | Projectile targeting | torch |
| `gradcheck_demo.kg` | Verify gradient correctness | torch |
| `compile_demo.kg` | Function compilation for speed | torch |

---

## Detailed Example Documentation

### basic_gradient.kg

**What it does:** Demonstrates fundamental gradient computation with the `:>` operator.

**Run:**
```bash
kgpy --backend torch basic_gradient.kg
```

**Expected output:**
```
Gradient of x^2 at x=3: 6.0
Gradient of x^3 at x=2: 12.0
...
```

**Key concepts:**
- Basic syntax: `f:>x` computes df/dx
- Works with polynomials, composed functions, arrays

---

### numeric_vs_autograd.kg

**What it does:** Compares the two gradient operators available in KlongPy.

**Run:**
```bash
kgpy --backend torch numeric_vs_autograd.kg
```

**Expected output:**
```
Numeric gradient (nabla): 5.999999...
Autograd gradient (:>): 6.0
```

**Key concepts:**
- `nabla` (`point nabla function`) - Always uses finite differences
- `:>` (`function:>point`) - Uses PyTorch autograd when available

---

### gradient_descent.kg

**What it does:** Demonstrates optimization algorithms using gradients.

**Run:**
```bash
kgpy --backend torch gradient_descent.kg
```

**Expected output:**
```
Minimizing f(x) = (x-5)^2
Initial x: 0.0
After 100 steps: x = 4.999...
```

**Key concepts:**
- Gradient descent loop: `x := x - lr * gradient`
- Rosenbrock function optimization
- Quadratic curve fitting

---

### linear_regression.kg

**What it does:** Trains a linear model y = wx + b using gradient descent.

**Run:**
```bash
kgpy --backend torch linear_regression.kg
```

**Expected output:**
```
Training linear regression...
Learned w: ~2.0, b: ~1.0
(True values: w=2, b=1)
```

**Key concepts:**
- Mean squared error loss
- Parameter updates via gradients
- Convergence monitoring

---

### neural_net.kg

**What it does:** Builds neural networks from scratch using autograd.

**Run:**
```bash
kgpy --backend torch neural_net.kg
```

**Expected output:**
```
Training AND gate perceptron...
Training sin(x) approximator...
Predictions vs actual sin values
```

**Key concepts:**
- Single neuron (perceptron) learning logic gates
- Hidden layer networks
- Sigmoid activation function

---

### portfolio_opt.kg

**What it does:** Optimizes portfolio weights to maximize Sharpe ratio using real stock data.

**Prerequisites:** `pip install yfinance`

**Run:**
```bash
kgpy --backend torch portfolio_opt.kg
```

**Expected output:**
```
Portfolio Optimization with KlongPy Autograd
==============================================

Fetching stock data from Yahoo Finance (AAPL, MSFT, GOOGL, JPM)...
Loaded 501 days of returns

Expected annual returns: [0.37 0.31 0.42 0.33]
Annual volatilities: [0.21 0.22 0.29 0.22]

Initial weights: [0.25 0.25 0.25 0.25]
Initial Sharpe: 3.03

Training for 200 epochs with lr=0.01
--------------------------------------------------
Epoch 0: Sharpe=3.04, weights=[0.26 0.25 0.24 0.25]
...
Epoch 150: Sharpe=3.08, weights=[0.31 0.23 0.19 0.26]

Optimization complete!
Final Sharpe: 3.09
Weight allocation:
  AAPL: 31%
  MSFT: 23%
  GOOGL: 19%
  JPM: 26%
```

**Key concepts:**
- Real market data from Yahoo Finance
- Sharpe ratio: return / volatility
- Constraint handling via penalty terms
- Annualization: volatility * sqrt(252)

---

### optimizer_demo.kg

**What it does:** Shows how to use custom optimizer classes (SGD, Adam).

**Run:**
```bash
kgpy --backend torch optimizer_demo.kg
```

**Expected output:**
```
Example 1: Minimize f(theta) = theta^2
Starting at theta = 10.0
Running 50 steps...
Final theta: 0.00014...

Example 2: Linear regression (y = 2x + 1)
Learned parameters:
  w = 1.99
  b = 1.05

Example 3: Manual gradient descent with multi-param gradients
Final: a=0.00007, c=0.00004
```

**Key concepts:**
- Importing Python optimizers via `.pyf()`
- SGD with momentum
- Adam optimizer
- Multi-parameter gradient syntax: `f:>[a b]`

---

### black_scholes_greeks.kg

**What it does:** Computes option Greeks (Delta, Vega, Theta, Rho) using autograd instead of manual derivation.

**Run:**
```bash
kgpy --backend torch black_scholes_greeks.kg
```

**Expected output:**
```
Black-Scholes Greeks via Autograd
==================================

Option parameters:
  Spot: 100.0
  Strike: 100.0
  Rate: 0.05
  Time: 1.0
  Vol: 0.2

Option Price: 10.45

Greeks computed via autograd:
(No manual derivation required!)

  Delta (dP/dS): 0.6368
  Vega (dP/dVol): 37.52
  Theta (-dP/dT): -6.41
  Rho (dP/dR): 53.23

Analytical values (for verification):
  Delta: 0.6368
  Vega: 37.52

Autograd computes exact derivatives automatically!
```

**Key concepts:**
- Greeks are derivatives of option price
- `delta::priceOfSpot:>spot` - no formula derivation needed
- Works for ANY pricing model (exotic options, stochastic vol, etc.)
- Uses backend `erf` for accurate normal CDF

**Why this matters:** Traditional quant work requires deriving Greek formulas by hand. With autograd, you just define the price function and differentiate automatically.

---

### black_scholes_live.kg

**What it does:** Fetches live AAPL option data and verifies Black-Scholes model against market prices.

**Prerequisites:** `pip install yfinance`

**Run:**
```bash
kgpy --backend torch black_scholes_live.kg
```

**Expected output:**
```
Black-Scholes Greeks - LIVE Market Verification
================================================

Fetching live AAPL option data from Yahoo Finance...

Live Option Parameters:
  Ticker: AAPL, Expiration: 2026-02-20
  Spot: $259.48
  Strike: $260.0
  Time to expiry: 0.055 years
  Implied Vol: 24.97%
  Risk-free Rate: 4.3%
  Dividend Yield: 0.5%

=== PRICE COMPARISON ===
  Model Price:  $6.06
  Market Mid:   $5.82
  Market Bid:   $5.70
  Market Ask:   $5.95

  Difference: $0.23
  Error: 4.0%

=== GREEKS (via Autograd) ===
  Delta: 0.51
  Vega (per 1% vol): 0.24
  Theta (per day): -0.17
  Rho (per 1%): 0.07

Small differences are expected due to:
  - Interest rate and dividend assumptions
  - Market data timing delays
  - Black-Scholes model limitations
```

**Key concepts:**
- Real-time option chain from Yahoo Finance
- Model validation against market prices
- Greeks interpretation:
  - Delta ~0.5 for at-the-money options
  - Theta is daily time decay
  - Vega is sensitivity to volatility

---

### differentiable_physics.kg

**What it does:** Uses gradient descent to find the optimal launch angle to hit a target. Demonstrates backpropagation through physics equations.

**Run:**
```bash
kgpy --backend torch differentiable_physics.kg
```

**Expected output:**
```
Differentiable Physics: Projectile Targeting
=============================================

Goal: Find the launch angle to hit a target at x=50
Method: Backpropagate through physics equations!

Target x: 50.0
Initial velocity: 25.0

Initial angle (rad): 0.5
Initial angle (deg): 28.65
Initial range: 53.61
Initial miss: 3.61

Optimizing launch angle via gradient descent...
(Gradients computed through physics equations!)

Epoch 0: angle=25.80 deg, range=49.93, miss=0.07
Epoch 20: angle=25.85 deg, range=50.0, miss=0.0
...

Optimization complete!
Final angle (deg): 25.85
Final range: 50.0
Final miss: 0.0

Analytical optimal angle (approx): 27.25

The gradient propagated through physics equations!
```

**Key concepts:**
- Projectile range formula: R = (v0^2 * sin(2*angle)) / g
- Loss = squared distance from target
- Gradient flows through physics equations
- Applications: robot control, game AI, scientific inverse problems

**Why this matters:** Instead of solving equations analytically, we can optimize initial conditions by differentiating through the physics simulation.

---

## Helper Files

### optimizers.py

Reusable optimizer classes for gradient descent:

```python
SGDOptimizer  # SGD with optional momentum
AdamOptimizer # Adam with adaptive learning rates
```

**Usage in Klong:**
```klong
.pyf("./optimizers.py";"SGDOptimizer")
opt::SGDOptimizer(["w" "b"];:{["lr" 0.1]})
{opt(loss)}'!100
```

### stock_data.py

Fetches stock returns from Yahoo Finance for portfolio optimization.

### live_options.py

Fetches live option chain data from Yahoo Finance for Black-Scholes verification.

---

## Using Autograd in Your Own Code

### Basic Gradient

```klong
f::{x^2}        :" Define function
f:>3            :" Gradient at x=3 -> 6.0
```

### Gradient Descent Pattern

```klong
theta::10.0
lr::0.1
loss::{(theta-5)^2}

:" Training step
step::{grad::loss:>theta; theta::theta-(lr*grad)}

:" Train
step'!100
.p(theta)       :" -> ~5.0
```

### Multi-Parameter Gradients

```klong
w::2.0
b::3.0
loss::{(w^2)+(b^2)}

:" Get gradients for both parameters
grads::loss:>[w b]    :" Returns [4.0, 6.0]

:" Update
w::w-(0.1*grads@0)
b::b-(0.1*grads@1)
```

---

## Performance Notes

With `--backend torch`:
- Exact gradients via PyTorch autograd
- GPU acceleration (CUDA/MPS) for large arrays
- Optimized tensor operations

Without torch:
- Falls back to numeric differentiation
- Uses finite differences: f'(x) ~ (f(x+e) - f(x-e)) / 2e
- Slower and less precise

---

### gradcheck_demo.kg

**What it does:** Demonstrates gradient verification using `.gradcheck()` to ensure autograd computes mathematically correct gradients.

**Run:**
```bash
kgpy --backend torch gradcheck_demo.kg
```

**Expected output:**
```
Gradient Verification Demo
==========================

Example 1: Verifying f(x) = x^2
--------------------------------
  Function: f(x) = x^2
  .gradcheck(f;3.0) = 1
  PASSED: Gradient is correct!

Example 2: Verifying p(x) = x^3 + 2x^2 - 5x + 3
...
All gradient checks passed!
```

**Key concepts:**
- `.gradcheck(fn;input)` returns 1 if gradients are correct
- Uses `torch.autograd.gradcheck` internally
- Compares autograd vs numeric differentiation
- Essential for verifying custom functions

---

### compile_demo.kg

**What it does:** Shows how to compile Klong functions for optimized execution using `torch.compile`.

**Run:**
```bash
kgpy --backend torch compile_demo.kg
```

**Expected output:**
```
Function Compilation Demo
=========================

Example 1: Compiling a simple function
---------------------------------------
  Original function: f(x) = x^2
  Compiled with: .compile(f;3.0)

  f(5.0) = 25.0
  cf(5.0) = 25.0
  Both give the same result!
...
```

**Key concepts:**
- `.compile(fn;input)` returns an optimized function
- Compiled functions run faster for complex computations
- First call has compilation overhead
- Best for functions called many times (training loops)

---

## Gradient Verification

Use `.gradcheck()` to verify your gradients are correct:

```klong
f::{x^2}
.gradcheck(f;3.0)            :" Returns 1 if correct

g::{+/x^2}
.gradcheck(g;[1.0 2.0 3.0])  :" Verifies vector gradients
```

This compares autograd gradients against numeric gradients using `torch.autograd.gradcheck`.

---

## Function Compilation

Compile functions for optimized execution:

```klong
f::{(x^3)+(2*x^2)+x}
cf::.compile(f;2.0)          :" Returns compiled function
cf(5.0)                       :" Faster execution
```

Export computation graphs for inspection:

```klong
info::.export(f;2.0;"model.pt2")
.p(info@"graph")              :" Print graph structure
```

---

## Troubleshooting

**"undefined: ..."** - Variable name conflicts with Klong reserved words. Rename variables to avoid `x`, `y`, `z` in outer scopes.

**MPS tensor errors** - Apple Silicon GPU tensors can't convert to numpy. Precompute constants or use CPU.

**yfinance errors** - Market may be closed or data unavailable. Try again during market hours.

**Gradient is NaN** - Learning rate too high or loss function has issues. Try smaller learning rate.

**torch.compile errors** - Requires C++ compiler. On macOS, install Xcode Command Line Tools.
