import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Title
st.title("🌸 Iris Flower Species Predictor")

# Load Iris data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
model = RandomForestClassifier()
model.fit(X, y)

# User Inputs
st.subheader("Enter Flower Measurements:")
sepal_length = st.slider("Sepal Length (cm)", float(X["sepal length (cm)"].min()), float(X["sepal length (cm)"].max()))
sepal_width = st.slider("Sepal Width (cm)", float(X["sepal width (cm)"].min()), float(X["sepal width (cm)"].max()))
petal_length = st.slider("Petal Length (cm)", float(X["petal length (cm)"].min()), float(X["petal length (cm)"].max()))
petal_width = st.slider("Petal Width (cm)", float(X["petal width (cm)"].min()), float(X["petal width (cm)"].max()))

# Prediction
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=iris.feature_names)
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)

# Show Results
st.subheader("Prediction:")
st.write(f"🌼 This flower is most likely a **{iris.target_names[prediction]}**.")

st.subheader("Prediction Probabilities:")
st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))
