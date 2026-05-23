AHHH yes — now we’re entering the real machine learning connection. Excellent direction.  
You’re asking about:  
Softmax in Multi-Class Logistic Regression  
and especially:  
Why the loss function changes  

This is one of the most important transitions in ML:  
Binary Logistic Regression  
→ becomes  
Multiclass Logistic Regression (Softmax Regression)  

Let’s build this very carefully from the ground up.  

## Big Picture First

We already know:  
Binary Logistic Regression  
handles:  
Yes/No  
0/1  
Positive/Negative  

using:  
Sigmoid  

But what if we have:  
Cat  
Dog  
Horse  
?  

Now there are:  
More than 2 classes  
Sigmoid alone is no longer enough.  

So we move to:  
Softmax Regression  
also called:  
Multinomial Logistic Regression  

Very important name.  

## Part 1 — Recall Binary Logistic Regression

In binary logistic regression:  
We compute:  
\( z = w^Tx+b \)  

Then apply sigmoid:  
\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

This gives:  
\[
P(y=1|x)
\]

Probability of positive class.  

### Why Sigmoid Works for Binary

Because binary classification only needs:  

| Class | Probability |
|------|-------------|
| Positive | p |
| Negative | 1-p |

Only ONE probability is enough.  

### Problem in Multi-Class

Suppose 3 classes:  
Cat  
Dog  
Horse  

Can sigmoid produce:  
3 probabilities  
summing to 1  
competing with each other  
?  

No.  

That’s the problem.  

## Part 2 — Why Independent Sigmoids Fail

Imagine using 3 sigmoids independently:  

| Class | Output |
|------|--------|
| Cat | 0.9 |
| Dog | 0.8 |
| Horse | 0.7 |

Total:  
\[
0.9+0.8+0.7=2.4
\]

Not a valid probability distribution.  

Also:  
classes are not competing  
model can strongly believe multiple classes simultaneously  

Bad for mutually exclusive classes.  

### Therefore We Need Softmax

Softmax ensures:  
All outputs between 0 and 1  
Total sum = 1  
Classes compete  

Perfect for multiclass classification.  

## Part 3 — Structure of Multiclass Logistic Regression

Suppose:  
Input features:  
\[
x=[x_1,x_2,x_3]
\]

and classes:  
Cat  
Dog  
Horse  

Instead of ONE weight vector:  
Binary logistic regression had:  
\[
w
\]

Now we need:  
One weight vector per class  

### Weight Matrix

We now use:  
\[
W
\]

Example:  
\[
W=
\begin{bmatrix}
--- \\
w_{cat} \\
--- \\
w_{dog} \\
--- \\
w_{horse}
\end{bmatrix}
\]

Each class learns its own scoring rule.  

## Part 4 — Logits in Softmax Regression

For each class we compute score:  
\[
z_k=w_k^Tx+b_k
\]

So:  

| Class | Score |
|------|-------|
| Cat | \(z_1\) |
| Dog | \(z_2\) |
| Horse | \(z_3\) |

These are called:  
logits  

Exactly like before.  

### Then Apply Softmax

Softmax converts logits into probabilities:  
\[
P(y=i|x)=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

Now probabilities:  
sum to 1  
compete properly  

## Part 5 — Numerical Example

Suppose logits:  

| Class | Logit |
|------|-------|
| Cat | 2 |
| Dog | 1 |
| Horse | 0 |

### Step 1 — Exponentials

\[
e^2=7.389
\]

\[
e^1=2.718
\]

\[
e^0=1
\]

### Step 2 — Sum

\[
7.389+2.718+1=11.107
\]

### Step 3 — Normalize

Cat:  
\[
7.389/11.107=0.665
\]

Dog:  
\[
2.718/11.107=0.245
\]

Horse:  
\[
1/11.107=0.09
\]

### Final Probabilities

| Class | Probability |
|------|-------------|
| Cat | 66.5% |
| Dog | 24.5% |
| Horse | 9% |

Prediction:  
Cat  
because highest probability.  

## Part 6 — NOW THE IMPORTANT PART

Why Loss Function Changes  

This is the heart of your question.  

### Binary Logistic Regression Loss

For binary classification:  
We used:  
Binary Cross Entropy  

Formula:  
\[
L=-(y\log(\hat y)+(1-y)\log(1-\hat y))
\]

Why?  
Because:  
only two classes exist  
output is single probability  

### Problem in Multi-Class

Now we have:  
multiple probabilities  
vector outputs  

Example:  
Prediction:  

| Class | Probability |
|------|-------------|
| Cat | 0.7 |
| Dog | 0.2 |
| Horse | 0.1 |

True class:  
Dog  

How do we compute loss now?  

Binary loss no longer fits naturally.  

## Part 7 — One-Hot Encoding

We represent true class as vector.  

Suppose true class = Dog.  
Then:  

| Class | Target |
|------|--------|
| Cat | 0 |
| Dog | 1 |
| Horse | 0 |

Vector:  
\[
y=[0,1,0]
\]

called:  
One-Hot Encoding  

## Part 8 — Multiclass Cross Entropy Loss

Now loss becomes:  
\[
L=-\sum_i y_i\log(\hat y_i)
\]

This is:  
Categorical Cross Entropy  
or:  
Softmax Loss  

### Why This Formula Works

Remember:  
Only one target entry is 1.  

Example:  
\[
y=[0,1,0]
\]

Prediction:  
\[
\hat y=[0.7,0.2,0.1]
\]

Plug in:  
\[
L=-(0\log0.7+1\log0.2+0\log0.1)
\]

Everything disappears except correct class:  
\[
L=-\log(0.2)
\]

### HUGE Intuition

The loss only cares about:  
probability assigned to correct class  

### If Correct Class Probability High

Example:  
\[
-\log(0.95)
\]

Very small loss.  
Good prediction.  

### If Correct Class Probability Low

Example:  
\[
-\log(0.01)
\]

Huge loss.  
Model punished heavily.  

## Part 9 — Why Logarithm?

The log gives beautiful behavior:  

| Probability | Loss |
|------|------|
| 1.0 | 0 |
| 0.9 | small |
| 0.5 | moderate |
| 0.01 | huge |

So:  
confident wrong predictions get heavily punished  

This is VERY important in learning.  

## Part 10 — Connection to Maximum Likelihood

Softmax + cross entropy comes from:  
Maximum Likelihood Estimation (MLE)  

The model tries to maximize:  
\[
P(\text{correct class})
\]

Taking negative log converts:  
multiplication → addition  
easier optimization  

This is deep statistical foundation.  

## Part 11 — Beautiful Gradient Simplification

This is one of the most elegant results in ML.  
Softmax + Cross Entropy derivative simplifies to:  
\[
\frac{\partial L}{\partial z}=\hat y-y
\]

Meaning:  
prediction − truth  

Very elegant.  
This simplicity is one reason neural networks train efficiently.  

## Part 12 — Binary Logistic Regression is Actually Special Case

Very important insight.  
Binary logistic regression is actually:  
a special case of softmax regression  
with:  
only 2 classes.  

Softmax generalizes logistic regression.  

## Part 13 — Full Pipeline

Binary Logistic Regression  
\[
z=w^Tx+b
\]

↓  
Sigmoid  
↓  
Binary Cross Entropy  

Multiclass Logistic Regression  
\[
z_k=w_k^Tx+b_k
\]

↓  
Softmax  
↓  
Categorical Cross Entropy  

## Final Deep Intuition

Softmax Regression is basically:  
“Logistic regression extended to multiple competing classes.”  

And the loss changes because:  
outputs become probability vectors  
targets become one-hot vectors  
learning now evaluates entire distributions rather than one binary probability.  

That’s the conceptual jump.  

## Next we can continue with:
- Deriving softmax loss mathematically step-by-step  
- Why derivative becomes (\(\hat y-y\))  
- Softmax regression from probability theory  
- Geometric interpretation of multiclass logistic regression  
- Difference between multinomial vs one-vs-rest logistic regression  
- Full NumPy implementation from scratch  
- Backpropagation through softmax  
- Decision boundaries in softmax regression  
- Information theory interpretation of cross entropy  
- Deep intuition of logits and why we don’t directly predict probabilities