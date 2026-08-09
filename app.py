import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.title("🎓 Smart Student Performance Prediction")

st.write("Enter student details to predict academic performance.")

# Load dataset
data = pd.read_csv("student_performance.csv")

# Features and target
X = data[["study_hours", "attendance", "previous_marks", "assignment_score"]]
y = data["performance"]

# Encode target
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train model
model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X, y_encoded)

# Student inputs
st.header("Enter Student Details")

study_hours = st.number_input(
    "Study Hours", min_value=0.0, max_value=24.0, value=5.0
)

attendance = st.number_input(
    "Attendance (%)", min_value=0.0, max_value=100.0, value=75.0
)

previous_marks = st.number_input(
    "Previous Marks", min_value=0.0, max_value=100.0, value=70.0
)

assignment_score = st.number_input(
    "Assignment Score", min_value=0.0, max_value=100.0, value=70.0
)

# Prediction button
if st.button("Predict Performance"):

    new_student = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "assignment_score": assignment_score
    }])

    prediction = model.predict(new_student)

    result = encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted Performance: {result}")

    st.write("### Student Details")
    st.write(f"Study Hours: {study_hours}")
    st.write(f"Attendance: {attendance}%")
    st.write(f"Previous Marks: {previous_marks}")
    st.write(f"Assignment Score: {assignment_score}")