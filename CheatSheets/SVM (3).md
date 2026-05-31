LET'S GOOOOOO 🚀🔥

Now we're entering the **actual optimization formulation** of SVM.

Everything so far was preparation.

Now we'll derive the famous constraint:

[
y_i(w^T x_i+b)\geq 1
]

from scratch.

---

# Lecture 3: Hard Margin SVM

## Our Goal

We already proved:

\text{Margin}=\frac{2}{|w|}

and therefore:

[
\text{maximize margin}
]

is equivalent to

[
\min ||w||
]

or more commonly

[
\min \frac12 ||w||^2
]

But we are missing something important.

---

Suppose I choose:

[
w=[0,0]
]

Then:

[
||w||=0
]

which is the smallest possible value!

Should SVM choose this?

Obviously not.

Why?

Because this doesn't classify anything.

So we need **constraints**.

---

# Constraint 1: Positive Class

Recall our support planes:

Positive support plane:

[
w^Tx+b=1
]

Negative support plane:

[
w^Tx+b=-1
]

---

For a positive sample:

[
y=+1
]

we want it to lie on or beyond the positive support plane.

So:

[
w^Tx+b \geq 1
]

---

Example

Suppose:

[
w^Tx+b=3
]

Then:

[
3 \geq 1
]

Good.

The point is correctly classified and outside the margin.

---

# Constraint 2: Negative Class

For negative samples:

[
y=-1
]

we want them on or beyond the negative support plane.

So:

[
w^Tx+b \leq -1
]

Example:

[
-5 \leq -1
]

Good.

---

# We Have Two Constraints

Positive class:

[
w^Tx+b \geq 1
]

Negative class:

[
w^Tx+b \leq -1
]

This works.

But mathematicians hate having two separate equations when one can do the job.

Let's combine them.

---

# Trick: Multiply by y

Remember:

[
y\in{-1,+1}
]

Only two possible values.

---

## Case 1: Positive Sample

[
y=+1
]

Constraint:

[
w^Tx+b \geq 1
]

Multiply by (+1):

[
(+1)(w^Tx+b)\geq 1
]

[
y(w^Tx+b)\geq 1
]

Looks nice.

---

## Case 2: Negative Sample

[
y=-1
]

Constraint:

[
w^Tx+b\leq -1
]

Multiply by (-1).

IMPORTANT:

When multiplying an inequality by a negative number, the sign flips.

So:

[
-(w^Tx+b)\geq 1
]

Since (y=-1):

[
y(w^Tx+b)\geq 1
]

WOW.

Same equation again.

---

# Unified Constraint

For every training example:

y_i(w^Tx_i+b)\geq1

This single equation handles BOTH classes.

---

# Let's Verify

Suppose:

[
y=+1
]

and

[
w^Tx+b=4
]

Then:

[
1\times4=4
]

[
4\geq1
]

Valid.

---

Suppose:

[
y=-1
]

and

[
w^Tx+b=-6
]

Then:

[
(-1)(-6)=6
]

[
6\geq1
]

Valid.

---

Suppose a point is misclassified.

[
y=-1
]

but

[
w^Tx+b=3
]

Then:

[
(-1)(3)=-3
]

[
-3\geq1
]

False.

Constraint violated.

Exactly what we want.

---

# The Complete Hard Margin Problem

Now we can finally write SVM mathematically.

Objective:

[
\min \frac12 ||w||^2
]

Subject to:

[
y_i(w^Tx_i+b)\geq1
]

for every training sample.

---

This is the famous Hard Margin SVM optimization problem:

[
\begin{aligned}
\min_{w,b}\quad & \frac12 ||w||^2 \
\text{subject to}\quad &
y_i(w^Tx_i+b)\ge1
\end{aligned}
]

---

# What Does It Mean Geometrically?

Think of it as:

### Objective

```text
Make ||w|| as small as possible
```

which means:

```text
Make margin as large as possible
```

---

### Constraint

```text
Don't let any point enter the margin
```

and

```text
Don't misclassify any point
```

---

# Why Is It Called Hard Margin?

Because the rules are strict.

Every single point must satisfy:

[
y_i(w^Tx_i+b)\ge1
]

No exceptions.

No mistakes.

No tolerance.

No mercy. 😎

---

# Problem With Real Data

Imagine this dataset:

```text
R R R R

      B

          B B B
```

One blue point accidentally appears inside the red region.

This happens all the time in real life:

* Measurement errors
* Noisy sensors
* Wrong labels
* Human mistakes

Now no line can perfectly separate the classes.

---

Hard Margin SVM says:

```text
I refuse.
```

No feasible solution exists.

The optimization breaks.

---

And that is exactly why researchers invented:

# Soft Margin SVM

which allows a few violations and introduces:

* Slack Variables ((\xi))
* Regularization Parameter ((C))
* Hinge Loss

These are the concepts used in real-world SVMs and are arguably more important than Hard Margin SVM.

We're now standing at the doorway of the version of SVM that libraries like `scikit-learn` actually use. 🔥

Next: **Soft Margin SVM — the birth of slack variables and the parameter (C).** 🚀
