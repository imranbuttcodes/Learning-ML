# From Perceptron to Logistic Regression
# The Complete Mathematical Journey
> ML Deep Theory Series | Imran's Study Notes

---

## The Timeline

```
1943 → McCulloch-Pitts Neuron    (first mathematical neuron — manual weights)
1958 → Perceptron                (first learning algorithm — auto weight update)
1969 → Perceptron Crisis         (can't solve XOR — AI Winter)
1970s→ Gradient Descent need     (need smooth functions for calculus)
1980s→ Sigmoid + MLE             (smooth activation + principled loss function)
Today→ Logistic Regression       (single neuron, sigmoid, gradient descent)
```

Each step was a direct solution to the previous step's problem.

---

# CHAPTER 1 — McCULLOCH-PITTS NEURON (1943)

---

## Inspiration — Biological Neuron

```
Biological neuron:

  Input signals (dendrites)
       │  │  │
       ↓  ↓  ↓
  ┌─────────────┐
  │  Cell body  │ ← sums all signals
  │   (soma)    │ ← if total > threshold → fires
  └──────┬──────┘
         ↓
    Output (axon)
```

McCulloch and Pitts modeled this mathematically in 1943.

## The Math Model

```
Inputs:  x₁, x₂, x₃
Weights: w₁, w₂, w₃  (importance of each input)

Step 1 — Weighted sum:
z = w₁x₁ + w₂x₂ + w₃x₃

Step 2 — Threshold:
output = 1  if z >= threshold
output = 0  if z <  threshold
```

**Fatal limitation:** Weights set MANUALLY by humans.
Model cannot learn from data on its own.

---

# CHAPTER 2 — THE PERCEPTRON (1958)

---

## Frank Rosenblatt's Innovation

Added automatic weight learning from data via error correction.

## Structure

```
x₁ ──w₁──┐
x₂ ──w₂──┤──→  z = Σwᵢxᵢ + b  ──→  step(z)  ──→  ŷ (0 or 1)
x₃ ──w₃──┘

b = bias term (shifts the threshold)
```

## The Step Function

```
step(z) = 1  if z >= 0
          0  if z <  0

Output
1 |     _________
  |    |
0 |____|
  |_____________
       z=0
Sharp jump — no smooth gradient anywhere
```

## The Learning Algorithm

```
For each training example (x, y):
  1. Calculate: ŷ = step(w·x + b)
  2. Error:     error = y - ŷ
  3. Update:    wᵢ = wᵢ + α × error × xᵢ
                b  = b  + α × error

α = learning rate (step size)
```

**Why it works:**

```
Predicted 0, should be 1 (error = +1):
wᵢ = wᵢ + α×(+1)×xᵢ → weights increase → z bigger next time → more likely 1

Predicted 1, should be 0 (error = -1):
wᵢ = wᵢ + α×(-1)×xᵢ → weights decrease → z smaller next time → more likely 0

Correct prediction (error = 0):
wᵢ = wᵢ + 0 → no change ✅
```

## Convergence Theorem

```
If data is LINEARLY SEPARABLE:
→ Perceptron WILL find correct weights
→ In finite number of steps
→ MATHEMATICALLY GUARANTEED

First provably correct learning algorithm in history.
```

## The Fatal Problem (1969)

Minsky & Papert proved Perceptron **cannot solve XOR**:

```
XOR:               Plot:
x₁  x₂  y          x₂
0   0    0          1 | ○(0,1)=1    ●(1,1)=0
0   1    1          0 | ●(0,0)=0    ○(1,0)=1
1   0    1            |_____________________ x₁
1   1    0

Can you draw ONE straight line separating ○ from ●?
NO. Impossible. XOR is not linearly separable.
→ Perceptron fails. AI Winter begins.
```

---

# CHAPTER 3 — THE GRADIENT DESCENT PROBLEM

---

## What is Gradient Descent?

```
Imagine blindfolded on a hilly landscape. Goal: reach lowest point.

Strategy:
1. Feel the slope (gradient)
2. Step in downhill direction
3. Repeat

Math:
w = w - α × dLoss/dw

dLoss/dw = gradient (slope of loss w.r.t. weight)
α         = learning rate (step size)
```

## Why Perceptron Can't Use It

```
Step function derivative:
d/dz step(z) = 0 everywhere (except undefined at z=0)

Output
1 |     _________
  |    |             ← gradient = 0 here
0 |____|             ← gradient = 0 here
  |_____________
  gradient=0   gradient=0   EVERYWHERE

No slope → gradient descent has no direction to move.
Gradient descent completely fails on step function.
```

**We need a function that is:**
- Smooth and differentiable everywhere
- Has meaningful gradients everywhere
- Still maps output to a reasonable range

---

# CHAPTER 4 — THE SIGMOID SOLUTION

---

## Replace Step With Sigmoid

```
σ(z) = 1 / (1 + e^(-z))

Step function:                 Sigmoid function:
1 |     _________              1 |          ___________
  |    |                         |        /
0 |____|                       0.5|------/
  |                              |    /
                               0 |___/_______________
                                     z=0

Sharp jump                     Smooth S-curve
Gradient = 0 everywhere        Gradient exists everywhere ✅
Can't use gradient descent     Can use gradient descent ✅
```

## Key Properties

```
z → -∞   →  σ(z) → 0     (definitely class 0)
z =  0   →  σ(z) = 0.5   (exactly on boundary, 50/50)
z → +∞   →  σ(z) → 1     (definitely class 1)
Always between 0 and 1     ✅
```

## The Sigmoid Derivative — Elegant Math

```
σ(z) = 1 / (1 + e^(-z))

Derivative:
dσ/dz = σ(z) × (1 - σ(z))

Derivation:
d/dz [1/(1+e^(-z))]
= e^(-z) / (1+e^(-z))²
= [1/(1+e^(-z))] × [e^(-z)/(1+e^(-z))]
= σ(z) × [1 - 1/(1+e^(-z))]
= σ(z) × (1 - σ(z))    ✅
```

**What this means:**

```
σ(z) = 0.5  → dσ/dz = 0.5×0.5 = 0.25   ← maximum gradient (uncertain)
σ(z) = 0.8  → dσ/dz = 0.8×0.2 = 0.16   ← moderate gradient
σ(z) → 0   → dσ/dz → 0                  ← gradient vanishes (very confident)
σ(z) → 1   → dσ/dz → 0                  ← gradient vanishes (very confident)

Model learns fast when uncertain, slows when confident.
This makes intuitive sense. ✅
```

---

# CHAPTER 5 — THE LOG LOSS DERIVATION

---

## Why Not MSE?

```
Problem 1 — Non-convex loss:
MSE + sigmoid = multiple local minima
Gradient descent gets stuck in wrong valley

Problem 2 — Weak gradients when wrong:
MSE gradient = (ŷ-y) × σ'(z) × x
σ'(z) → 0 when model is very wrong (σ near 0 or 1)
Learning slows exactly when it should speed up ❌
```

## Maximum Likelihood Estimation — Where Log Loss Comes From

MLE principle: **find weights that make the observed training data most probable.**

### Step 1 — Probability of One Example

```
P(y=1 | x) = σ(z) = ŷ
P(y=0 | x) = 1 - σ(z) = 1 - ŷ

Both cases in one formula:
P(y | x) = ŷʸ × (1-ŷ)^(1-y)

Check:
y=1: ŷ¹ × (1-ŷ)⁰ = ŷ        ✅
y=0: ŷ⁰ × (1-ŷ)¹ = (1-ŷ)    ✅
```

### Step 2 — Joint Likelihood (All Examples)

```
L(w) = Π ᵢ P(yᵢ | xᵢ)
     = Π ᵢ ŷᵢʸⁱ × (1-ŷᵢ)^(1-yᵢ)

(product assumes examples are independent)
```

### Step 3 — Log Likelihood

```
log converts products to sums → easier math + numerically stable

log L(w) = Σᵢ [yᵢ × log(ŷᵢ) + (1-yᵢ) × log(1-ŷᵢ)]
```

### Step 4 — From Maximize to Minimize

```
We want to MAXIMIZE log likelihood.
Optimization traditionally MINIMIZES.
Negate it:

Minimize: -log L(w) = -Σᵢ [yᵢ×log(ŷᵢ) + (1-yᵢ)×log(1-ŷᵢ)]

Average over n examples:

Log Loss = -(1/n) Σᵢ [yᵢ×log(ŷᵢ) + (1-yᵢ)×log(1-ŷᵢ)]
```

**Log Loss was DERIVED from first principles — not invented arbitrarily.**

## Why Log Loss Penalizes Correctly

```
y=1, ŷ=0.95  (confident, correct):
Loss = -log(0.95) = 0.051    ← tiny ✅

y=1, ŷ=0.05  (confident, WRONG):
Loss = -log(0.05) = 2.996    ← huge ❌

y=0, ŷ=0.05  (confident, correct):
Loss = -log(0.95) = 0.051    ← tiny ✅

y=0, ŷ=0.95  (confident, WRONG):
Loss = -log(0.05) = 2.996    ← huge ❌

Penalty grows EXPONENTIALLY as model gets more confidently wrong.
```

```
-log(x) curve:
x=0.99 → 0.01   (almost certain, correct → tiny penalty)
x=0.50 → 0.69   (uncertain)
x=0.10 → 2.30   (wrong)
x=0.01 → 4.61   (confidently wrong → huge penalty)
```

---

# CHAPTER 6 — GRADIENT DESCENT ON LOG LOSS

---

## Computing the Gradient

```
Log Loss: L = -(1/n) Σ [y×log(σ(z)) + (1-y)×log(1-σ(z))]
where z = w·x + b

Need: ∂L/∂w  using chain rule

∂L/∂w = ∂L/∂ŷ × ∂ŷ/∂z × ∂z/∂w
```

### Step 1 — ∂L/∂ŷ

```
L = -[y×log(ŷ) + (1-y)×log(1-ŷ)]
∂L/∂ŷ = -y/ŷ + (1-y)/(1-ŷ)
```

### Step 2 — ∂ŷ/∂z (sigmoid derivative)

```
∂σ/∂z = σ(z)×(1-σ(z)) = ŷ×(1-ŷ)
```

### Step 3 — ∂z/∂w

```
z = w·x + b
∂z/∂w = x
```

### Combining — The Beautiful Simplification

```
∂L/∂w = [-y/ŷ + (1-y)/(1-ŷ)] × ŷ×(1-ŷ) × x

Simplify the first two terms:
[-y/ŷ + (1-y)/(1-ŷ)] × ŷ×(1-ŷ)
= -y×(1-ŷ) + (1-y)×ŷ
= -y + y×ŷ + ŷ - y×ŷ
= ŷ - y

Therefore:
∂L/∂w = (ŷ - y) × x    ← prediction error × input
∂L/∂b = (ŷ - y)        ← prediction error
```

**Stunningly elegant — gradient = (predicted - actual) × input.**
Same intuitive form as linear regression.

## The Weight Update Rule

```
wⱼ = wⱼ - α × (1/n) Σᵢ (ŷᵢ - yᵢ) × xᵢⱼ
b  = b  - α × (1/n) Σᵢ (ŷᵢ - yᵢ)

α = learning rate
```

## Step-by-Step Example

```
Features: study_hours=6, sleep_hours=7, y=1 (passed)
Start: w₁=0, w₂=0, b=0, α=0.1

Forward pass:
z  = 0×6 + 0×7 + 0 = 0
ŷ  = σ(0) = 0.5

Loss:
L  = -log(0.5) = 0.693

Gradients:
∂L/∂w₁ = (0.5-1)×6 = -3.0
∂L/∂w₂ = (0.5-1)×7 = -3.5
∂L/∂b  = (0.5-1)   = -0.5

Update:
w₁ = 0 - 0.1×(-3.0) = 0.30
w₂ = 0 - 0.1×(-3.5) = 0.35
b  = 0 - 0.1×(-0.5) = 0.05

New prediction:
z  = 0.30×6 + 0.35×7 + 0.05 = 4.30
ŷ  = σ(4.30) = 0.987   ← much closer to 1 ✅
```

## Gradient Descent Variants

```
Batch GD:       use ALL examples per update → smooth but slow
SGD:            use ONE example per update  → fast but noisy
Mini-batch GD:  use 32-128 examples         → standard in practice
```

---

# CHAPTER 7 — THE COMPLETE PICTURE

---

## Everything Connected

```
McCulloch-Pitts (1943):
  z = Σ wᵢxᵢ
  output = step(z)
  Problem: weights manual, step has no gradient

Perceptron (1958):
  Same + automatic weight update via error correction
  Problem: step function → no gradient → can't stack layers → XOR fails

Sigmoid (1980s):
  Replace step(z) with σ(z) = 1/(1+e^(-z))
  Smooth, differentiable everywhere
  σ'(z) = σ(z)(1-σ(z))

MLE → Log Loss:
  Maximize P(data|weights)
  → maximize log likelihood
  → minimize negative log likelihood
  → Log Loss = -(1/n)Σ[y×log(ŷ)+(1-y)×log(1-ŷ)]

Gradient of Log Loss:
  ∂L/∂w = (ŷ-y) × x   ← elegant, simple
  ∂L/∂b = (ŷ-y)

Gradient Descent:
  w = w - α × ∂L/∂w
  Repeat until convergence

= Logistic Regression
```

## Final Comparison

```
                Perceptron              Logistic Regression
                ──────────              ───────────────────
Year            1958                    1980s
Activation      Step function           Sigmoid
Output          Hard 0 or 1             Probability (0,1)
Gradient        None                    Exists everywhere
Learning        Error correction        Gradient descent on Log Loss
Loss function   None formal             Log Loss (from MLE)
Convergence     Linearly sep. only      Always (convex loss)
Probabilistic   ❌                      ✅
Stackable       ❌ (no gradient)        ✅ → Neural Networks
```

## The Path to Neural Networks

```
Logistic Regression = 1 neuron with sigmoid

Stack N neurons in layers = Neural Network

Input → [n₁][n₂][n₃]  →  [n₄][n₅]  →  [n₆]  → Output
         Hidden Layer 1    Hidden L2    Output

Each neuron:
z = w·x + b
output = activation(z)    ← sigmoid, ReLU, tanh...

Backpropagation = gradient descent through all layers
using chain rule — same math, just deeper.

Everything in deep learning traces back
to this exact derivation.
```

---

## Summary of Key Formulas

```
Forward pass:
z  = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
ŷ  = σ(z) = 1/(1+e^(-z))

Loss:
L  = -(1/n) Σ [y×log(ŷ) + (1-y)×log(1-ŷ)]

Sigmoid derivative:
dσ/dz = σ(z) × (1-σ(z))

Gradient (via chain rule):
∂L/∂w = (ŷ - y) × x
∂L/∂b = (ŷ - y)

Weight update:
w = w - α × ∂L/∂w
b = b - α × ∂L/∂b
```