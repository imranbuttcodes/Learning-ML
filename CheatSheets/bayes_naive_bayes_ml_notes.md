# 🧠 Bayes' Theorem & Naive Bayes — Complete ML Guide (FULL)

# 📌 1. Introduction to Probability

Probability measures how likely an event is.

0 <= P(A) <= 1

- 0 → impossible event  
- 1 → certain event  

Examples:
- Coin flip → 0.5  
- Dice roll (6) → 1/6  

In ML, probability = model confidence

---

# 📌 2. Conditional Probability

P(A|B) = P(A ∩ B) / P(B)

Example:
Dice = {1,2,3,4,5,6}  
B = even → {2,4,6}

P(4|even) = 1/3

---

# 📌 3. Bayes’ Theorem

P(A|B) = (P(B|A) * P(A)) / P(B)

Prior = P(A)
Likelihood = P(B|A)
Evidence = P(B)
Posterior = P(A|B)

---

# 📌 4. Naive Bayes Classifier

P(C|X) ∝ P(C) * Π P(xi|C)

Assumes independence between features.

---

# 📌 5. Gaussian Naive Bayes (Core Idea)

Used for continuous data.

Steps:
1. Compute mean
2. Compute std
3. Use Gaussian probability
4. Multiply likelihoods
5. Choose max class

---

# 📌 6. Gaussian Naive Bayes from Scratch

import math

X = [
    [98, 72],
    [99, 75],
    [97, 70],
    [102, 95],
    [103, 100],
    [101, 92]
]

y = [0, 0, 0, 1, 1, 1]

def separate(X, y):
    data = {}
    for i in range(len(X)):
        label = y[i]
        if label not in data:
            data[label] = []
        data[label].append(X[i])
    return data

def mean(nums):
    return sum(nums) / len(nums)

def std(nums):
    avg = mean(nums)
    return (sum((x - avg) ** 2 for x in nums) / len(nums)) ** 0.5

def summarize(data):
    summaries = []
    for col in zip(*data):
        col = list(col)
        summaries.append({
            "mean": mean(col),
            "std": std(col)
        })
    return summaries

def gaussian(x, mean, std):
    import math
    if std == 0:
        return 1e-9
    return (1 / ((2 * 3.14159) ** 0.5 * std)) * math.exp(-((x - mean) ** 2) / (2 * std ** 2))

def train(X, y):
    separated = separate(X, y)
    model = {}
    total = len(X)
    for label, rows in separated.items():
        model[label] = {
            "prior": len(rows) / total,
            "summaries": summarize(rows)
        }
    return model

def predict(model, x):
    results = {}
    for label, info in model.items():
        prob = info["prior"]
        for i, feature in enumerate(info["summaries"]):
            prob *= gaussian(x[i], feature["mean"], feature["std"])
        results[label] = prob
    return max(results, key=results.get), results

model = train(X, y)

prediction, probs = predict(model, [100, 90])

print(prediction)
print(probs)

---

# 📌 7. Multinomial Naive Bayes

Used for text classification:

P(C) * Π P(word|C)^count

---

# 📌 8. Laplace Smoothing

(count + 1) / (total + V)

Prevents zero probability issue.

---

# 📌 9. scikit-learn Example

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

texts = ["free money win", "win free prize", "meeting tomorrow", "project discussion"]
y = [1,1,0,0]

vec = CountVectorizer()
X = vec.fit_transform(texts)

model = MultinomialNB()
model.fit(X, y)

print(model.predict(vec.transform(["free prize"])))

---

# 📌 10. Final Summary

Bayes theorem → update belief using evidence  
Naive Bayes → probabilistic classifier  
Gaussian NB → continuous data  
Multinomial NB → text data  
Bernoulli NB → binary data
