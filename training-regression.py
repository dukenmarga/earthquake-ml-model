import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import PolynomialFeatures

filename = "dataset.csv"
earthquake_df = pd.read_csv(filename, low_memory=False)

# Scale the features (if needed, optional)
# Normal scaling: [x - min(x)] / [max(x) - min(x)]
scale_col = ["PGA", "PWaveFreq", "Mass", "Stiffness", "Damping", "NaturalFreq"]
for col in scale_col:
    earthquake_df[col] = (earthquake_df[col] - earthquake_df[col].min()) / (
        earthquake_df[col].max() - earthquake_df[col].min()
    )

# CORRELATION
# Correlation > 0.7 paired with Overall will be considered as important feature
limit = 0.0
numerical_columns = earthquake_df.select_dtypes(include=[np.number])
feat = []
for col in earthquake_df:
    # Skip
    if col in ["MaxDisplacement"]:
        continue

    # Append feature that pass the correlation limit
    corr = earthquake_df["MaxDisplacement"].corr(earthquake_df[col])
    if abs(corr) > limit:
        print(f"{col}: {corr:.2f}")
        feat.append(col)
print(f"Significant features: {feat}\n")

# Identify the features you will use in your model
# For clarity, ml feat is explicitly defined here (exluding Overall and Best Overall)
ml_features = [
    "PGA",
    "PWaveFreq",
    "Mass",
    "Stiffness",
    "Damping",
    "NaturalFreq",
]
ml_target = ["MaxDisplacement"]
# ml_features = feat

# Pair plot
sns.pairplot(
    earthquake_df[
        [
            "PGA",
            "PWaveFreq",
            "Mass",
            "Stiffness",
            "Damping",
            "NaturalFreq",
            "MaxDisplacement",
        ]
    ]
)
# plt.show()

# Split data into training set and test set
X: pd.DataFrame = earthquake_df[ml_features]
y = earthquake_df[ml_target]  # type: ignore

# Separate data: 20% Testing, 80% Training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # type: ignore

# Create polynomial features
quadratic = PolynomialFeatures(degree=5)
X_train_quad = quadratic.fit_transform(X_train)  # type: ignore
X_test_quad = quadratic.transform(X_test)  # type: ignore

regression = LinearRegression()
regression.fit(X_train_quad, y_train)
y_pred: NDArray[np.float64] = regression.predict(X_test_quad)

# Using Random Forest to predict
# rf_model = RandomForestRegressor()
# rf_model.fit(X_train, y_train)
# y_pred: NDArray[np.float64] = rf_model.predict(X_test)

# for i in range(3):
#     tree = rf_model.estimators_[i]
#     dot_data = export_graphviz(
#         tree,
#         feature_names=X_train.columns,
#         filled=True,
#         max_depth=2,
#         impurity=False,
#         proportion=True,
#     )
#     graph = graphviz.Source(dot_data)
#     display(graph)

# Testing accuracy and F1 score

# Calculate MSE and R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# ml_accuracy = accuracy_score(y_test, y_pred)
# ml_f1_score = float(f1_score(y_true=y_test, y_pred=y_pred, average="weighted"))

# print(f"Accuracy: {ml_accuracy}")
# print(f"F1 Score: {ml_f1_score}")
