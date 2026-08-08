import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
data = pd.read_csv("student_performance.csv")

print("First 5 records:")
print(data.head())

# 2. Separate features and target
X = data[["study_hours", "attendance", "previous_marks", "assignment_score"]]
y = data["performance"]

# 3. Encode target labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 5. Train the machine learning model
model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=encoder.inverse_transform([0, 1, 2]),
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Predict performance for a new student
new_student = pd.DataFrame([{
    "study_hours": 7,
    "attendance": 90,
    "previous_marks": 80,
    "assignment_score": 85
}])

prediction = model.predict(new_student)
predicted_label = encoder.inverse_transform(prediction)[0]

print("\nNew Student Prediction:")
print("Study Hours: 7")
print("Attendance: 90%")
print("Previous Marks: 80")
print("Assignment Score: 85")
print("Predicted Performance:", predicted_label)

# 9. Create a graph
plt.figure(figsize=(8, 5))
plt.scatter(data["study_hours"], data["previous_marks"])
plt.xlabel("Study Hours")
plt.ylabel("Previous Marks")
plt.title("Study Hours vs Previous Marks")
plt.grid(True)
plt.tight_layout()
plt.savefig("results.png")
plt.show()
