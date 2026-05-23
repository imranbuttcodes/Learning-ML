We’ll manually do:  
FULL Gradient Descent Update  
for Softmax Regression  
using the SAME fruit example.

We’ll go slowly and deeply.

## Goal

We want to see:
- Forward pass
- Softmax probabilities
- Cross entropy loss
- Gradients
- Weight updates
- How learning actually happens

## Our Tiny Dataset

Input feature:
\[
x=5
\]

True class:
Mango

Classes:
Apple
Mango
Lemon

## Initial Parameters

We use SAME weights as before.

| Class | Weight (w) | Bias (b) |
|------|------------|----------|
| Apple | 0.3 | 0.1 |
| Mango | 0.8 | -1 |
| Lemon | -0.5 | 1 |

## Step 1 — Forward Pass

Compute logits.

Formula:
\[
z_k=w_kx+b_k
\]

Input:
\[
x=5
\]

Apple
\[
z_a=0.3(5)+0.1
\]
\[
=1.6
\]

Mango
\[
z_m=0.8(5)-1
\]
\[
=3
\]

Lemon
\[
z_l=-0.5(5)+1
\]
\[
=-1.5
\]

### Logits

| Class | Logit |
|------|-------|
| Apple | 1.6 |
| Mango | 3 |
| Lemon | -1.5 |

## Step 2 — Softmax

Formula:
\[
\hat y_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

Exponentials:
\[
e^{1.6}=4.95
\]
\[
e^3=20.09
\]
\[
e^{-1.5}=0.223
\]

Sum:
\[
25.263
\]

### Probabilities

Apple:
\[
\hat y_a=4.95/25.263=0.196
\]

Mango:
\[
\hat y_m=20.09/25.263=0.795
\]

Lemon:
\[
\hat y_l=0.223/25.263=0.009
\]

### Final Prediction

| Class | Probability |
|------|-------------|
| Apple | 0.196 |
| Mango | 0.795 |
| Lemon | 0.009 |

## Step 3 — One-Hot Target

True class = Mango.

So:

| Class | Target |
|------|--------|
| Apple | 0 |
| Mango | 1 |
| Lemon | 0 |

Vector:
\[
y=[0,1,0]
\]

## Step 4 — Cross Entropy Loss

Formula:
\[
L=-\sum_i y_i\log(\hat y_i)
\]

Substitute:
\[
L=-\log(0.795)
\]

\[
=0.229
\]

Good prediction → small loss.

## NOW THE IMPORTANT PART  
## BACKPROPAGATION

This is where learning happens.

## Step 5 — Gradient of Softmax + Cross Entropy

The magical result:
\[
\frac{\partial L}{\partial z_i}=\hat y_i-y_i
\]

Meaning:
prediction − target

This is the error signal.

### Compute Gradients for Logits

Apple Gradient
\[
0.196-0
\]
\[
=0.196
\]

Mango Gradient
\[
0.795-1
\]
\[
=-0.205
\]

Lemon Gradient
\[
0.009-0
\]
\[
=0.009
\]

### Gradient Table

| Class | Gradient |
|------|----------|
| Apple | 0.196 |
| Mango | -0.205 |
| Lemon | 0.009 |

## HUGE Intuition

Positive gradient:
score too high
reduce it

Negative gradient:
score too low
increase it

So:

| Class | Meaning |
|------|---------|
| Apple | decrease score |
| Mango | increase score |
| Lemon | slightly decrease |

This is learning.

## Step 6 — Gradient w.r.t Weights

Remember:
\[
z_k=w_kx+b_k
\]

Using chain rule:
\[
\frac{\partial L}{\partial w_k}=\frac{\partial L}{\partial z_k}\cdot x
\]

Input:
\[
x=5
\]

Apple Weight Gradient
\[
0.196 \times 5
\]
\[
=0.98
\]

Mango Weight Gradient
\[
-0.205 \times 5
\]
\[
=-1.025
\]

Lemon Weight Gradient
\[
0.009 \times 5
\]
\[
=0.045
\]

### Weight Gradient Table

| Class | Weight Gradient |
|------|-----------------|
| Apple | 0.98 |
| Mango | -1.025 |
| Lemon | 0.045 |

## Step 7 — Bias Gradients

Bias derivative is simpler.
Because:
\[
\frac{\partial z}{\partial b}=1
\]

So:
\[
\frac{\partial L}{\partial b_k}=\hat y_k-y_k
\]

### Bias gradients:

| Class | Bias Gradient |
|------|---------------|
| Apple | 0.196 |
| Mango | -0.205 |
| Lemon | 0.009 |

## Step 8 — Gradient Descent Update

Update rule:
\[
w_{new}=w_{old}-\eta\frac{\partial L}{\partial w}
\]

where:
\[
\eta=\text{learning rate}
\]

Let:
\[
\eta=0.1
\]

### Update Apple Weight

Old:
\[
0.3
\]

Gradient:
\[
0.98
\]

Update:
\[
0.3-0.1(0.98)
\]
\[
=0.202
\]

### Update Mango Weight

Old:
\[
0.8
\]

Gradient:
\[
-1.025
\]

Update:
\[
0.8-0.1(-1.025)
\]
\[
=0.9025
\]

### Update Lemon Weight

Old:
\[
-0.5
\]

Gradient:
\[
0.045
\]

Update:
\[
-0.5-0.1(0.045)
\]
\[
=-0.5045
\]

### Updated Weights

| Class | Old Weight | New Weight |
|------|------------|------------|
| Apple | 0.3 | 0.202 |
| Mango | 0.8 | 0.9025 |
| Lemon | -0.5 | -0.5045 |

## Step 9 — Bias Updates

Same formula.

Apple Bias
\[
0.1-0.1(0.196)
\]
\[
=0.0804
\]

Mango Bias
\[
-1-0.1(-0.205)
\]
\[
=-0.9795
\]

Lemon Bias
\[
1-0.1(0.009)
\]
\[
=0.9991
\]

## HUGE Intuition About Learning

Notice what happened.

### Mango
Correct class.
Its weight INCREASED:
\[
0.8 \to 0.9025
\]

Meaning:
future Mango scores become larger
Model becomes more confident.

### Apple
Wrong class.
Weight DECREASED:
\[
0.3 \to 0.202
\]

Meaning:
Apple score reduces next time

### Lemon
Already tiny probability.
Small adjustment only.

THIS IS LEARNING

Gradient descent literally says:
“Increase correct class score.
Decrease incorrect class scores.”

Repeated thousands of times.

## Step 10 — What Happens Next Epoch?

If we run forward pass again:
- Mango probability becomes even larger
- loss decreases further
- Eventually model converges.

## Full Pipeline Recap

Forward Pass
\[
z=wx+b
\]

↓  
Softmax  
probabilities  

↓  
Cross Entropy  
loss  

↓  
Gradients  
\[
\hat y-y
\]

↓  
Update weights  

↓  
Better predictions  

↓  
Repeat  

## Final Deep Intuition

Softmax Regression learns by:
- comparing predicted probabilities against truth
- measuring error using cross entropy
- pushing correct class logits upward
- pushing wrong class logits downward

That’s the soul of multiclass classification.

## Next we can continue with:
- FULL derivative proof of softmax + cross entropy
- Matrix/vector form of all calculations
- Multiple training examples manually
- Batch gradient descent
- NumPy implementation from scratch
- Visualizing decision boundaries
- Why softmax gradients are coupled
- Stochastic gradient descent intuition
- Backpropagation through neural networks
- Geometric interpretation of softmax learning