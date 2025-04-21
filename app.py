import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Streamlit page configuration
st.set_page_config(page_title="ML Classification App", layout="wide")
st.title("Machine Learning Classification App")

# 1. Dataset Upload
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Select features and target
    all_columns = df.columns.tolist()
    features = st.multiselect("Select feature columns", all_columns, default=all_columns[:-1])
    target = st.selectbox("Select target column", all_columns)

    if features and target:
        X = df[features]
        y = df[target]

        # 2. Data Split and Scaling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 3. Model Selection
        st.header("Model Training and Evaluation")
        model_choice = st.selectbox("Choose a model", ("Logistic Regression", "Decision Tree", "KNN"))

        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=200)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            # KNN default
            n_neighbors = st.sidebar.slider("Number of Neighbors (K)", 1, 20, 3)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        # 4. Evaluation Metrics
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {acc:.2f}")

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=np.unique(y), yticklabels=np.unique(y))
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)
    else:
        st.warning("Please select feature and target columns.")
else:
    st.info("Awaiting CSV file to be uploaded.")
