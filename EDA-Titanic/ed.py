# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Step 1: Load the Titanic dataset
# You can load it from Seaborn's built-in datasets
df = sns.load_dataset('titanic')

# Step 2: Data Preprocessing
# Fill missing values in 'age' with the median age
df['age'].fillna(df['age'].median(), inplace=True)

# Fill missing values in 'embarked' with the mode (most frequent) value
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Drop the rows where 'deck' is missing, for simplicity
df.drop(['deck'], axis=1, inplace=True)

# Drop the 'who', 'adult_male', 'alive', and 'class' columns (redundant)
df.drop(['who', 'adult_male', 'alive', 'class'], axis=1, inplace=True)

# Convert categorical variables into dummy/indicator variables
df = pd.get_dummies(df, columns=['sex', 'embarked', 'embark_town'], drop_first=True)

# Drop the remaining 'sibsp' and 'parch' columns as they are not useful for this analysis
df.drop(['sibsp', 'parch'], axis=1, inplace=True)

# Drop rows where 'fare' or 'embarked' are missing
df.dropna(subset=['fare'], inplace=True)

# Step 3: Exploratory Data Analysis (EDA)

# 3.1. Distribution of Age and Fare
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=30, kde=True, color='blue')
plt.title('Distribution of Age')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['fare'], bins=30, kde=True, color='green')
plt.title('Distribution of Fare')
plt.show()

# 3.2. Survival Rate by Gender
plt.figure(figsize=(8, 5))
sns.barplot(x='sex_male', y='survived', data=df)
plt.title('Survival Rate by Gender')
plt.xticks([0, 1], ['Female', 'Male'])
plt.show()

# 3.3. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Step 4: Build the Machine Learning Model

# Define the target variable (y) and feature variables (X)
X = df.drop('survived', axis=1)
y = df['survived']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Step 5: Evaluate the Model

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# ROC Curve
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Step 6: Visualize Feature Importance
# Logistic Regression's coefficients represent feature importance
feature_importance = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Feature Importance')
plt.show()
