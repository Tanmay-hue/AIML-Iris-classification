import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set Streamlit page configuration
st.set_page_config(page_title="Iris Classification", layout="wide")

st.title("Iris Flower Classification App")

# 1. Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

st.header("Dataset Overview")
st.write("The Iris dataset contains 150 samples with the following features:")
st.write(X.head())

# 2. Data Exploration (EDA)
st.header("Exploratory Data Analysis")
st.subheader("Summary Statistics")
st.write(X.describe())

st.subheader("Pair Plot")
# Create pairplot using seaborn
pairplot_fig = sns.pairplot(pd.concat([X, y], axis=1), hue='species')
st.pyplot(pairplot_fig.fig)

# 3. Data Preprocessing
st.header("Data Preprocessing")
# Split data into training and testing sets (80:20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.write("Data has been split and scaled.")

# 4. Model Training and Evaluation
st.header("Model Training and Evaluation")

model_choice = st.selectbox("Choose a model", ("Logistic Regression", "Decision Tree", "KNN"))

if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
elif model_choice == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)  # No scaling needed for decision trees
    y_pred = model.predict(X_test)
elif model_choice == "KNN":
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy:** {acc:.2f}")

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# 5. Confusion Matrix Visualization
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=iris.target_names, yticklabels=iris.target_names)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
st.pyplot(fig)

# 6. Optional: Hyperparameter Tuning for KNN
if model_choice == "KNN":
    st.subheader("Hyperparameter Tuning for KNN")
    n_neighbors = st.slider("Select number of neighbors (n)", 1, 20, 3)
    knn_tuned = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_tuned.fit(X_train_scaled, y_train)
    y_pred_tuned = knn_tuned.predict(X_test_scaled)
    st.write(f"**Tuned KNN Accuracy:** {accuracy_score(y_test, y_pred_tuned):.2f}")
    st.text(classification_report(y_test, y_pred_tuned))
