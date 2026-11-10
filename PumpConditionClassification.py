
"""
Practical Engineering Case Study:
Pump Condition Classification

Algorithms:
1. SVM
2. Naive Bayes
3. Decision Tree
4. Random Forest
5. AdaBoost

Class labels:
0 = Normal
1 = Warning
2 = Fault
"""

# Importing the required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)


# ============================================================
# PART 1. CREATE DATASET (30 RECORDS)
# ============================================================

data = [
    # ID, Vibration, RPM, Temperature, Pressure, Operating Hours, Condition

    # Normal = 0
    [1,  1.1, 1480, 42, 4.8,  850, 0],
    [2,  1.4, 1490, 45, 4.7, 1100, 0],
    [3,  1.6, 1475, 48, 4.9, 1350, 0],
    [4,  1.3, 1500, 44, 5.0,  950, 0],
    [5,  1.8, 1460, 50, 4.6, 1650, 0],
    [6,  1.5, 1510, 46, 4.8, 1200, 0],
    [7,  2.0, 1450, 52, 4.5, 1900, 0],
    [8,  1.2, 1495, 43, 4.9, 1000, 0],
    [9,  1.7, 1485, 49, 4.7, 1500, 0],
    [10, 1.9, 1470, 51, 4.6, 1800, 0],

    # Warning = 1
    [11, 3.2, 1430, 61, 4.1, 2800, 1],
    [12, 3.8, 1410, 65, 3.9, 3200, 1],
    [13, 4.1, 1400, 68, 3.8, 3500, 1],
    [14, 3.5, 1440, 63, 4.0, 3000, 1],
    [15, 4.4, 1390, 70, 3.6, 3900, 1],
    [16, 3.7, 1420, 66, 3.9, 3300, 1],
    [17, 4.8, 1370, 73, 3.5, 4200, 1],
    [18, 3.4, 1450, 60, 4.2, 2700, 1],
    [19, 4.2, 1380, 71, 3.7, 3800, 1],
    [20, 4.6, 1360, 75, 3.4, 4500, 1],

    # Fault = 2
    [21, 6.1, 1320, 82, 2.9, 5200, 2],
    [22, 6.8, 1290, 88, 2.6, 6100, 2],
    [23, 7.2, 1270, 92, 2.4, 6800, 2],
    [24, 5.9, 1340, 80, 3.0, 4900, 2],
    [25, 7.8, 1250, 96, 2.2, 7400, 2],
    [26, 6.5, 1300, 86, 2.7, 5900, 2],
    [27, 8.2, 1230, 101, 1.9, 8200, 2],
    [28, 5.6, 1350, 78, 3.1, 4700, 2],
    [29, 7.5, 1260, 94, 2.3, 7100, 2],
    [30, 8.5, 1210, 105, 1.8, 8700, 2],
]

columns = [
    "Pump_ID",
    "Vibration_mm_s",
    "Speed_RPM",
    "Temperature_C",
    "Pressure_bar",
    "Operating_Hours",
    "Pump_Condition"
]

df = pd.DataFrame(data, columns=columns)

print("\nPUMP DATASET")
print(df.to_string(index=False))

print("\nClass Distribution:")
print(df["Pump_Condition"].value_counts().sort_index().rename(
    index={0: "Normal", 1: "Warning", 2: "Fault"}
))


# Separating input features and output class

feature_cols = [
    "Vibration_mm_s",
    "Speed_RPM",
    "Temperature_C",
    "Pressure_bar",
    "Operating_Hours"
]

X = df[feature_cols]
y = df["Pump_Condition"]

class_names = ["Normal", "Warning", "Fault"]


# Visualizing the dataset

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Temperature_C",
    y="Vibration_mm_s",
    hue="Pump_Condition",
    style="Pump_Condition",
    s=100
)

plt.title("Pump Temperature vs Vibration")
plt.xlabel("Temperature (°C)")
plt.ylabel("Vibration (mm/s)")
plt.legend(title="0=Normal, 1=Warning, 2=Fault")
plt.tight_layout()
plt.show()


# Correlation between the features

plt.figure(figsize=(9, 6))

sns.heatmap(
    df[feature_cols].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Pump Features")
plt.tight_layout()
plt.show()


# ============================================================
# PART 2. SPLIT THE DATASET INTO TRAINING AND TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# PART 3. CREATE THE FIVE CLASSIFICATION MODELS
# ============================================================

# 1. Support Vector Machine

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale"))
])


# 2. Naive Bayes

naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])


# 3. Decision Tree

decision_tree_model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)


# 4. Random Forest

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42
)


# 5. AdaBoost

adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)


# Putting all models together

models = {
    "SVM": svm_model,
    "Naive Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model
}


# ============================================================
# PART 4. TRAIN AND EVALUATE THE MODELS
# ============================================================

results = []
predictions = {}

for name, model in models.items():

    print("\n" + "=" * 55)
    print(name.upper())
    print("=" * 55)

    # Training the model
    model.fit(X_train, y_train)

    # Making predictions
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Fault precision and recall
    fault_precision = precision_score(
        y_test, y_pred,
        labels=[2],
        average="macro",
        zero_division=0
    )

    fault_recall = recall_score(
        y_test, y_pred,
        labels=[2],
        average="macro",
        zero_division=0
    )

    # Macro F1 score
    macro_f1 = f1_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    # Saving the results
    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Fault Precision": fault_precision,
        "Fault Recall": fault_recall,
        "Macro F1": macro_f1
    })

    print(f"Accuracy        : {accuracy:.3f}")
    print(f"Fault Precision : {fault_precision:.3f}")
    print(f"Fault Recall    : {fault_recall:.3f}")
    print(f"Macro F1-score  : {macro_f1:.3f}")

    # Showing classification report
    print("\nClassification Report:")

    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=class_names,
        zero_division=0
    ))

    # Drawing confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        display_labels=class_names,
        cmap="Blues",
        colorbar=False,
        ax=ax
    )

    ax.set_title(name + " Confusion Matrix")

    plt.tight_layout()
    plt.show()


# ============================================================
# PART 5. COMPARE THE FIVE ALGORITHMS
# ============================================================

results_df = pd.DataFrame(results).set_index("Algorithm")

print("\nMODEL COMPARISON")
print(results_df.round(3))

# Displaying the comparison using a bar chart

results_df.plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1.05)
)

plt.title("Performance Comparison of Classification Algorithms")
plt.xlabel("Algorithms")
plt.ylabel("Scores")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# ============================================================
# PART 6. FEATURE IMPORTANCE
# ============================================================

tree_models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

for name, model in tree_models.items():

    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    print("\n", name, "Feature Importance:")
    print(importance.round(3))

    plt.figure(figsize=(7, 4))

    plt.barh(importance.index, importance.values)

    plt.title(name + " Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()


# Displaying the Decision Tree

plt.figure(figsize=(18, 9))

plot_tree(
    decision_tree_model,
    feature_names=feature_cols,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree for Pump Classification")
plt.tight_layout()
plt.show()


# ============================================================
# PART 7. TEST A NEW PUMP OBSERVATION
# ============================================================

new_pump = pd.DataFrame([{
    "Vibration_mm_s": 7.2,
    "Speed_RPM": 1280,
    "Temperature_C": 92,
    "Pressure_bar": 2.4,
    "Operating_Hours": 6800
}])

print("\nNEW PUMP OBSERVATION")
print(new_pump)

condition_names = {
    0: "Normal",
    1: "Warning",
    2: "Fault"
}

print("\nPREDICTION RESULTS:")

for name, model in models.items():

    prediction = int(model.predict(new_pump)[0])

    print(name, ":", condition_names[prediction])


# ============================================================
# PART 8. FINAL OBSERVATION
# ============================================================

print("\nFINAL OBSERVATION")

print("""
Normal:
The pump is operating under normal conditions.

Warning:
The pump shows some abnormal conditions and needs inspection.

Fault:
The pump shows serious abnormal conditions and needs
engineering inspection.

The model predictions are only for assisting engineers.
They should not replace actual pump inspection and
industrial safety procedures.

Group Members
Dagim Demissew .... WOUR/0528/16
Hayat Shekur .... WOUR/4025/16
Mena Mulugeta .... WOUR/1320/16
""")

print("\nEND OF CLASSWORK")
