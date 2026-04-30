# 🧠 Machine Learning Evaluation Metrics

Evaluation metrics help measure how well a machine learning model performs after training. They answer two key questions: **How wrong is the model?** and **How well does it match reality?**

***

## 🚀 Regression Metrics

Regression models predict continuous values such as house prices, marks, sales, or temperature.

### 1. Mean Absolute Error (MAE)

$$
\text{MAE} = \frac{1}{n} \sum |y - \hat{y}|
$$

**Meaning:**
- Average of the absolute errors.
- Ignores whether the prediction was above or below the actual value.

**Intuition:**
- “On average, how far off is the model?”

***

### 2. Mean Squared Error (MSE)

$$
\text{MSE} = \frac{1}{n} \sum (y - \hat{y})^2
$$

**Meaning:**
- Squares each error before averaging.
- Gives more penalty to large mistakes.

**Intuition:**
- “How bad are the model’s errors overall?”

***

### 3. Root Mean Squared Error (RMSE)

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum (y - \hat{y})^2}
$$

**Meaning:**
- Square root of MSE.
- Returns the error in the original unit of the target variable.

**Intuition:**
- “What is the typical prediction error size?”

***

### 4. R² Score

$$
R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}
$$

**Meaning:**
- Shows how much variance in the target is explained by the model.

**Range:**

| Value | Meaning |
|-------|---------|
| 1.0 | Perfect model |
| 0.0 | No improvement over baseline |
| Less than 0 | Worse than predicting the average |

***

## 🐍 Python Example

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_true = [50, 55, 65, 70]
y_pred = [48, 57, 63, 72]

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
```

***

## 📌 When to Use Each Metric

| Metric | Best Use |
|--------|----------|
| **MAE** | General understanding of average error |
| **MSE** | When large errors should be punished more |
| **RMSE** | Real-world interpretation in original units |
| **R²** | Comparing overall model quality |

***

## 🔍 Classification Metrics

For classification models such as Logistic Regression, common evaluation metrics are:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

***

## 🎯 Big Picture

Evaluation metrics help decide whether a model is useful, whether it needs improvement, and which model performs better on the same problem.

### Quick Recall

- **MAE** → average mistake
- **MSE** → penalized mistake
- **RMSE** → error in real units
- **R²** → overall quality score

***

## 🚀 Next Learning Path

After understanding these metrics, the natural next topics are:

- Logistic Regression
- Classification metrics in depth
- End-to-end machine learning projects
- Model improvement and tuning