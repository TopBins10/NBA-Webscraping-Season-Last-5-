import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Load and prepare data
training = pd.read_excel('TrainingDataLebron.xlsx')
scaler = StandardScaler()
scaled_features = scaler.fit_transform(training.drop(columns=['Game Date']))
features = scaled_features
labels = training['REB'].values

# Convert to PyTorch tensors
features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32)

# Define the model
model = nn.Sequential(
    nn.Linear(features.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)  # Outputs a single value per input
)

# Define loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(1000):
    # Shuffle and split the dataset for each epoch
    X_train, X_test, y_train, y_test = train_test_split(features_tensor, labels_tensor, test_size=0.4, shuffle=True)

    # Create data loaders
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1)

    # Train the model
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

    # Evaluate the model
model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for inputs, targets in test_loader:
        outputs = model(inputs)
        # Flatten the outputs to match the targets
        predictions.extend(outputs.view(-1).numpy())
        actuals.extend(targets.numpy())

# Print predictions vs actual values for this epoch
print(f"Epoch {epoch+1} Predictions vs Actual Values:")
for pred, actual in zip(predictions, actuals):
    print(f"Predicted: {pred:.2f}, Actual: {actual}")

accData = pd.read_excel('seasonStats.xlsx')

# Drop the 'Game Date' and '+/-' columns
accData = accData.drop(columns=['Game Date', '+/-'])

# Save the modified DataFrame to a new Excel file

accData.to_excel('seasonStats.xlsx', index=False)

# Assuming accData is loaded as a pandas DataFrame
# Check for non-numeric data
print(accData.dtypes)

# Check for missing values
print(accData.isnull().sum())

# Normalize the data (use the same scaler object used for training data if possible)
accData_scaled = scaler.transform(accData)  # Replace 'scaler' with the scaler instance used earlier

# Convert to tensor
accData_tensor = torch.tensor(accData_scaled, dtype=torch.float32)

# Continue with the model evaluation as you have in your code


# Make sure your model is in evaluation mode
model.eval()

# Disable gradient calculations as they are not needed for inference
with torch.no_grad():
    # Pass the data through the model
    predictions = model(accData_tensor)

# Convert predictions to a numpy array if needed
predictions_np = predictions.numpy()