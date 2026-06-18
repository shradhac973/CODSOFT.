# Credit Card Fraud Detection Project

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE

# Loading the dataset
df = pd.read_csv("creditcard.csv")

# Understanding the size of the dataset
print("Dataset Shape:", df.shape)

# Checking how many genuine and fraud transactions are present
print("\nTransaction Distribution:")
print(df["Class"].value_counts())

# Visualizing the class distribution
plt.figure(figsize=(6,4))
df["Class"].value_counts().plot(kind="bar")
plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Class")
plt.ylabel("Number of Transactions")
plt.show()

# Separating features and target variable
X = df.drop("Class", axis=1)
y = df["Class"]

# Scaling Amount and Time columns
# This helps the model learn more effectively
scaler = StandardScaler()

X["Amount"] = scaler.fit_transform(X[["Amount"]])
X["Time"] = scaler.fit_transform(X[["Time"]])

# Splitting the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# The dataset contains very few fraud cases.
# Using SMOTE to balance the classes by generating synthetic fraud samples.
print("\nBefore SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

# Training the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_balanced, y_train_balanced)

# Predicting transaction classes on test data
y_pred = model.predict(X_test)

# Evaluating model performance
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Finding the most important features used by the model
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

# Visualizing top important features
plt.figure(figsize=(8,5))
plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Top 10 Important Features")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

print("\nFraud Detection Model Completed Successfully!")
