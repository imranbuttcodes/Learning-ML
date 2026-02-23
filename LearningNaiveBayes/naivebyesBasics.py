"""

---

## **1️⃣ Gaussian Naive Bayes (Titanic)**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch']
df = df.drop(['Name','Ticket','Cabin','PassengerId','Fare','Parch','SibSp'], axis=1)
df.fillna({'Age': df['Age'].mean(), 'Embarked': df['Embarked'].mode()[0]}, inplace=True)

# Separate numerical and categorical
num_cols = ['Pclass','Age','FamilySize']
cat_cols = ['Sex','Embarked']

# One-hot encode categorical
encoder = OneHotEncoder(sparse_output=False)
X_cat = encoder.fit_transform(df[cat_cols])
X_num = df[num_cols].values

# Scale numerical features
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Combine features
X = np.hstack([X_cat, X_num_scaled])
y = df['Survived'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Gaussian NB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print("Gaussian NB Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

✅ This is **Gaussian NB** — perfect for numeric features like Age or FamilySize.

---

## **2️⃣ Multinomial Naive Bayes (Text Example)**

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Simple text dataset
docs = ["buy cheap meds", "free offer just now", "meeting schedule today", "project report submission"]
labels = [1, 1, 0, 0]  # 1=spam, 0=not spam

# Convert to counts
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(docs)

# Train Multinomial NB
mnb = MultinomialNB()
mnb.fit(X_counts, labels)

# Predict new text
new_docs = ["free meds now", "project meeting"]
X_new = vectorizer.transform(new_docs)
pred = mnb.predict(X_new)

print("Multinomial NB Predictions:", pred)  # 1=spam, 0=not spam
```

✅ Multinomial NB works on **count-based features** (how many times words appear).

---

## **3️⃣ Bernoulli Naive Bayes (Binary Features)**

```python
from sklearn.naive_bayes import BernoulliNB

# Binary presence/absence
docs_bin = ["buy cheap meds", "free offer", "meeting today", "report submission"]
labels_bin = [1,1,0,0]

# Convert to binary features (1=word present, 0=absent)
vectorizer_bin = CountVectorizer(binary=True)
X_bin = vectorizer_bin.fit_transform(docs_bin)

# Train Bernoulli NB
bnb = BernoulliNB()
bnb.fit(X_bin, labels_bin)

# Predict new text
new_bin_docs = ["free meds", "project meeting"]
X_new_bin = vectorizer_bin.transform(new_bin_docs)
pred_bin = bnb.predict(X_new_bin)

print("Bernoulli NB Predictions:", pred_bin)  # 1=spam, 0=not spam
```

✅ Bernoulli NB works on **binary features**, e.g., “word present or not,” “yes/no features,” etc.

---

### 🔑 Takeaways:

1. **Gaussian NB** → continuous numbers (Age, Salary, Height…)
2. **Multinomial NB** → counts (text word frequency, click counts)
3. **Bernoulli NB** → binary (word present/absent, yes/no, 0/1)

---



"""


# 1️⃣ Gaussian Naive Bayes (Numeric Data)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Simple numeric dataset
# Features: Marks
# Labels: Pass(1)/Fail(0)
data = pd.DataFrame({
    'Marks': [70, 80, 40, 50],
    'Label': [1, 1, 0, 0]
})

X = data[['Marks']].values
y = data['Label'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Gaussian NB model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predictions
y_pred = gnb.predict(X_test)
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 2️⃣ Multinomial Naive Bayes (Text Data)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample Emails
emails = ["buy buy", "buy", "meeting meeting", "meeting"]
labels = [1, 1, 0, 0]  # 1 = Spam, 0 = NotSpam

# Convert text to word counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=42)

# Multinomial NB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

y_pred = mnb.predict(X_test)
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# 3️⃣ Bernoulli Naive Bayes (Text Binary Features)


from sklearn.naive_bayes import BernoulliNB

# Same text, but Bernoulli looks at word presence
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(emails)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=42)

# Bernoulli NB
bnb = BernoulliNB()
bnb.fit(X_train, y_train)

y_pred = bnb.predict(X_test)
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))