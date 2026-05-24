import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
n_samples = 1000

income = np.random.randint(2000, 20000, n_samples)
debt = np.random.randint(0, 15000, n_samples)
payment_history = np.random.randint(0, 2, n_samples)
age = np.random.randint(18, 70, n_samples)
loan_amount = np.random.randint(1000, 20000, n_samples)

credit_score = (
    (income - debt/2 + payment_history*5000 - loan_amount/3) > 5000
).astype(int)

df = pd.DataFrame({
    'income': income,
    'debt': debt,
    'payment_history': payment_history,
    'age': age,
    'loan_amount': loan_amount,
    'creditworthy': credit_score
})

print("Dataset sample:")
print(df.head())

X = df.drop('creditworthy', axis=1)
y = df['creditworthy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_pred)
    }

results_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:")
print(results_df)

best_model = RandomForestClassifier()
best_model.fit(X_train_scaled, y_train)

import joblib
joblib.dump(best_model, "credit_model.pkl")   
joblib.dump(scaler, "scaler.pkl")

print(" MODEL SAVED SUCCESFULLY ")

y_pred_best = best_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

y_pred_proba = best_model.predict_proba(X_test_scaled)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="Random Forest")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
