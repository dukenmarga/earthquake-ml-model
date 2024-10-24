import graphviz
import joblib
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from huggingface_hub import HfApi, HfFolder, create_repo, login
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,  # type: ignore
    f1_score,  # type: ignore
    precision_score,  # type: ignore
    recall_score,  # type: ignore
)
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.tree import export_graphviz

# Note:
# Make sure dataset.csv and dataset-for-testing.csv is in the same folder
# We separate the dataset into training and test set

# TRAINING

filename = "dataset.csv"
earthquake_df = pd.read_csv(filename, low_memory=False)

# Scale the features (if needed, optional)
# Normal scaling: [x - min(x)] / [max(x) - min(x)]
scale_col = ["Mass", "Stiffness"]
for col in scale_col:
    earthquake_df[col] = (earthquake_df[col] - earthquake_df[col].min()) / (
        earthquake_df[col].max() - earthquake_df[col].min()
    )


def action_category(x: np.float64) -> float:
    # if max displacement > 1 cm -> 1: Moderate/Severe earthquake, go outside the building
    if x > 0.01:
        return 1
    # else -> 0: Minor earthquake, can stay inside the building
    else:
        return 0


earthquake_df["Run"] = earthquake_df["MaxDisplacement"].apply(action_category)

# Correlation
numerical_columns = earthquake_df.select_dtypes(include=[np.number])
for col in earthquake_df:
    # Skip
    if col in ["Run"]:
        continue

    # Append feature that pass the correlation limit
    corr = earthquake_df["MaxDisplacement"].corr(earthquake_df[col])
    print(f"{col}: {corr:.2f}")

# Identify the features you will use in your model
# Others are comment out: low correlation number
ml_features = [
    "PGA",
    # "PWaveFreq",
    # "Mass",
    # "Stiffness",
    # "Damping",
    "NaturalFreq",
]
ml_target = ["Run"]

# Pair plot
sns.pairplot(
    earthquake_df[
        [
            "PGA",
            # "PWaveFreq",
            # "Mass",
            # "Stiffness",
            # "Damping",
            # "NaturalFreq",
            "MaxDisplacement",
        ]
    ]
)
# plt.show()
_ = plt

# Split data into training set and test set
X: pd.DataFrame = earthquake_df[ml_features]
y = earthquake_df[ml_target]  # type: ignore

# Separate data: 20% Testing, 80% Training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)  # type: ignore

# Using Random Forest to predict
rf_model = RandomForestClassifier(max_depth=8, criterion="gini", n_estimators=200)
rf_model.fit(X_train, y_train)


# TESTING
# To use different set of earthquakes
filename = "dataset-fixed-testing.csv"
earthquake_testing_df = pd.read_csv(filename, low_memory=False)

# Scale the features (if needed, optional)
# Normal scaling: [x - min(x)] / [max(x) - min(x)]
scale_col = ["Mass", "Stiffness"]
for col in scale_col:
    earthquake_testing_df[col] = (
        earthquake_testing_df[col] - earthquake_testing_df[col].min()
    ) / (earthquake_testing_df[col].max() - earthquake_testing_df[col].min())


earthquake_testing_df["Run"] = earthquake_testing_df["MaxDisplacement"].apply(
    action_category
)

# Split data into training set and test set
X: pd.DataFrame = earthquake_testing_df[ml_features]
y = earthquake_testing_df[ml_target]  # type: ignore

# Separate data: 20% Testing, 80% Training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95)  # type: ignore
y_pred: NDArray[np.float64] = rf_model.predict(X_test)

for i in range(3):
    tree = rf_model.estimators_[i]
    dot_data = export_graphviz(
        tree,
        feature_names=X_train.columns,
        filled=True,
        max_depth=2,
        impurity=False,
        proportion=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render()

# Testing accuracy and F1 score
ml_accuracy = accuracy_score(y_test, y_pred)
ml_f1_score = float(f1_score(y_test, y_pred, average="weighted"))
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")


print(f"Accuracy: {ml_accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {ml_f1_score}")

# Save the trained model to a file
joblib.dump(rf_model, "earthquake_model.joblib")
_ = joblib
print("Model saved as 'earthquake_model.joblib'")

# Push to Huggingface Hub
login()
create_repo(repo_id="dukenmarga/earthquake-model", private=False)


# Set repository details
model_name = "earthquake-model"  # Your repository name
model_path = "earthquake_model.joblib"  # Path to your model file

# Get your access token (it should be automatically available after login)
token = HfFolder.get_token()

# Upload the model file
api = HfApi()
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="earthquake_model.joblib",
    repo_id=f"dukenmarga/{model_name}",
    repo_type="model",
    token=token,
)
