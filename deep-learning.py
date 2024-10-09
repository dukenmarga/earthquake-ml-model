import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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
    # if max displacement > 5 cm -> 2: Severe earthquake, go outside the building
    if x > 0.05:
        return 2
    # if max displacement > 1 cm -> 1: Moderate earthquake, prepare for potential evacuation
    elif x > 0.01:
        return 1
    # if max displacement < 1 cm -> 0: Minor earthquake, can stay inside the building
    else:
        return 0


earthquake_df["Run"] = earthquake_df["MaxDisplacement"].apply(action_category)

# CORRELATION
# Correlation > 0.7 paired with Overall will be considered as important feature
limit = 0.0
numerical_columns = earthquake_df.select_dtypes(include=[np.number])
feat = []
for col in earthquake_df:
    # Skip
    if col in ["Run"]:
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
    # "PWaveFreq",
    # "Mass",
    # "Stiffness",
    # "Damping",
    "NaturalFreq",
]
ml_target = ["Run"]

X = earthquake_df[ml_features]
y = earthquake_df[ml_target]

# Convert diagnosis string to 1 (M) and 0 (B)
# y.loc[:, 'Diagnosis'] = y['Diagnosis'].map({'M': 1, 'B': 0})
# y = y.astype(int)

# Convert from Pandas to Pytorch
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)

print(f"Feature shape: {X_tensor.shape}")

# Split data into training and testing randomly
indices = torch.randperm(X_tensor.size(0))
x_train = torch.index_select(X_tensor, dim=0, index=indices[80:])
y_train = torch.index_select(y_tensor, dim=0, index=indices[80:])
x_test = torch.index_select(X_tensor, dim=0, index=indices[:80])
y_test = torch.index_select(y_tensor, dim=0, index=indices[:80])
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")


# Define model
class CancerClassifier(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer_1 = torch.nn.Linear(input_size, 100)
        self.layer_2 = torch.nn.Linear(100, 200)
        self.layer_3 = torch.nn.Linear(200, 400)
        self.layer_4 = torch.nn.Linear(400, 200)
        self.layer_5 = torch.nn.Linear(200, 100)
        self.layer_6 = torch.nn.Linear(100, 10)
        self.layer_7 = torch.nn.Linear(10, 1)

        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer_1(x))
        x = self.activation(self.layer_2(x))
        x = self.activation(self.layer_3(x))
        x = self.activation(self.layer_4(x))
        x = self.activation(self.layer_5(x))
        x = self.activation(self.layer_6(x))
        return self.layer_7(x)


# Get total feature and use this as input size
input_size = X.shape[1]  # it has 30 features

# Define loss function
# BCEWithLogitsLoss => BCE + Sigmoid
loss_fn = torch.nn.BCEWithLogitsLoss()

# Instantiate the model
model = CancerClassifier(input_size)

# Define optimizer
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Write your code here for Model Training here
dl_loss_value = 1

# define the iteration
num_epochs = 1500

training_losses = []
testing_losses = []

# create the training loop
for epoch in range(num_epochs):
    # Train mode
    model.train()

    # Feed Forward
    prediction = model(x_train)

    # Calculate loss
    loss = loss_fn(prediction, y_train)

    # Backward propagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss for every epoch
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.inference_mode():
            test_prediction = model(x_test)
            test_loss = loss_fn(test_prediction, y_test)

            training_losses.append(loss.data.numpy())
            testing_losses.append(test_loss.data.numpy())
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

dl_loss_value = loss.item()

plt.plot(training_losses, label="Training loss")
plt.plot(testing_losses, label="Testing loss")
plt.legend()
plt.show()

print("Training complete!")

# # TESTING
# filename = "dataset-fixed-testing.csv"
# earthquake_testing_df = pd.read_csv(filename, low_memory=False)

# # Scale the features (if needed, optional)
# # Normal scaling: [x - min(x)] / [max(x) - min(x)]
# scale_col = ["Mass", "Stiffness"]
# for col in scale_col:
#     earthquake_testing_df[col] = (
#         earthquake_testing_df[col] - earthquake_testing_df[col].min()
#     ) / (earthquake_testing_df[col].max() - earthquake_testing_df[col].min())


# earthquake_testing_df["Run"] = earthquake_testing_df["MaxDisplacement"].apply(
#     action_category
# )

# # Split data into training set and test set
# X: pd.DataFrame = earthquake_testing_df[ml_features]
# y = earthquake_testing_df[ml_target]  # type: ignore

# # Separate data: 20% Testing, 80% Training set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95)  # type: ignore
# y_pred: NDArray[np.float64] = rf_model.predict(X_test)
# print(X_test)

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
#     graph.render()

# # Testing accuracy and F1 score
# ml_accuracy = accuracy_score(y_test, y_pred)
# ml_f1_score = float(f1_score(y_true=y_test, y_pred=y_pred, average="weighted"))

# print(f"Accuracy: {ml_accuracy}")
# print(f"F1 Score: {ml_f1_score}")

# # Save the trained model to a file
# joblib.dump(rf_model, "earthquake_model.joblib")
# print("Model saved as 'earthquake_model.joblib'")
