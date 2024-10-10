import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

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
        return 1
    # if max displacement > 1 cm -> 1: Moderate earthquake, prepare for potential evacuation
    # elif x > 0.01:
    #     return 1
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

y_train = y_train.squeeze().long()
y_test = y_test.squeeze().long()


# Check available resources
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Available: {device}")

# Run model from CUDA if available
if torch.cuda.is_available():
    model.cuda()


x_train.to(device)
y_train.to(device)
x_test.to(device)
y_test.to(device)


# Define model
class EarthquakeClassifier(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer_1 = torch.nn.Linear(input_size, 100)
        self.layer_2 = torch.nn.Linear(100, 1000)
        self.layer_3 = torch.nn.Linear(1000, 100)
        self.layer_4 = torch.nn.Linear(100, 2)

        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer_1(x))
        x = self.activation(self.layer_2(x))
        x = self.activation(self.layer_3(x))
        return self.layer_4(x)


# Get total feature and use this as input size
input_size = X.shape[1]  # it has 30 features

# Define loss function
# BCEWithLogitsLoss => BCE + Sigmoid
loss_fn = torch.nn.CrossEntropyLoss()

# Instantiate the model
model = EarthquakeClassifier(input_size)

# Define optimizer
learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Write your code here for Model Training here
dl_loss_value = 1

# define the iteration
num_epochs = 100


training_losses = []
testing_losses = []
print(y_train.shape)


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

# TESTING
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

y_train = y_train.squeeze().long()
y_test = y_test.squeeze().long()

# Testing accuracy and F1 score
model.eval()
y_pred_logits = model(x_train)
y_pred = torch.argmax(y_pred_logits, dim=1)  # Get the index of the max logit

print(f"y_train: {y_train}")
print(f"y_pred: {y_pred}")

ml_accuracy = accuracy_score(y_train.detach().numpy(), y_pred.detach().numpy())
ml_f1_score = float(
    f1_score(
        y_true=y_train.detach().numpy(),
        y_pred=y_pred.detach().numpy(),
        average="weighted",
    )
)

print(f"Accuracy: {ml_accuracy}")
print(f"F1 Score: {ml_f1_score}")

# # Save the trained model to a file
joblib.dump(model, "earthquake_model_DL.joblib")
print("Model saved as 'earthquake_model_DL.joblib'")
