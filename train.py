import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fairlearn.datasets import fetch_adult

# ==========================================
# 1. Load and Prepare the Dataset
# ==========================================
print("Loading UCI Adult dataset...")
data = fetch_adult(as_frame=True)
X = data.data
y = data.target

# Convert target to binary (0 for <=50K, 1 for >50K)
y = (y == '>50K').astype(int)

# Isolate sensitive features for Fairlearn evaluation later
sensitive_features = X[['sex', 'race']].copy()

# ==========================================
# 2. Feature Engineering & Preprocessing
# ==========================================
# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['category', 'object']).columns

# One-hot encode categorical features (drops first to avoid multicollinearity)
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split data (Train: 80%, Test: 20%)
# We must split the sensitive features alongside X and y to keep indices aligned
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    X_encoded, y, sensitive_features, test_size=0.2, random_state=42, stratify=y
)

# Standardize numeric features based on training data
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Convert everything to float32 (standard for PyTorch NN weights)
X_train_np = X_train.astype(np.float32).values
y_train_np = y_train.astype(np.float32).values
X_test_np = X_test.astype(np.float32).values
y_test_np = y_test.astype(np.float32).values

# ==========================================
# 3. PyTorch Dataset and DataLoader
# ==========================================
class AdultDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features)
        self.y = torch.tensor(targets).unsqueeze(1) # Reshape to [batch_size, 1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create Dataset objects
train_dataset = AdultDataset(X_train_np, y_train_np)
test_dataset = AdultDataset(X_test_np, y_test_np)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training batches: {len(train_loader)}")
print(f"Feature dimension size: {X_train_np.shape[1]}")

import torch.nn as nn
import torch.optim as optim

# ==========================================
# 4. Define the Neural Network Architecture
# ==========================================
class IncomeClassifier(nn.Module):
    def __init__(self, input_dim):
        super(IncomeClassifier, self).__init__()
        # Input layer to first hidden layer (64 neurons)
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        
        # First hidden layer to second hidden layer (32 neurons)
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        
        # Second hidden layer to output layer (1 neuron)
        self.output_layer = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid() # Squashes output to a probability between 0 and 1

    def forward(self, x):
        # This defines how data flows through the network
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.sigmoid(self.output_layer(x))
        return x

# ==========================================
# 5. Initialize Model, Loss, and Optimizer
# ==========================================
# We use the feature dimension size from Step 1
input_dimension = X_train_np.shape[1]

# Instantiate the model
model = IncomeClassifier(input_dim=input_dimension)

# Define the Loss function and the Optimizer
criterion = nn.BCELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n--- Step 2 Complete ---")
print("Model Architecture:")
print(model)

# ==========================================
# 6. The Training Loop
# ==========================================
epochs = 20
print(f"\n--- Step 3: Training the Model for {epochs} Epochs ---")

for epoch in range(epochs):
    model.train() # Set the model to training mode
    epoch_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        # 1. Zero the gradients
        optimizer.zero_grad()
        
        # 2. Forward pass
        predictions = model(batch_X)
        
        # 3. Calculate loss
        loss = criterion(predictions, batch_y)
        
        # 4. Backward pass
        loss.backward()
        
        # 5. Update weights
        optimizer.step()
        
        epoch_loss += loss.item()
        
    # Calculate average loss for the epoch
    avg_loss = epoch_loss / len(train_loader)
    
    # Print progress every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Average Loss: {avg_loss:.4f}")

print("Training Complete!")

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ==========================================
# 7. Standard Performance Evaluation
# ==========================================
print("\n--- Step 4: Evaluating Model Performance ---")
model.eval() # Set model to evaluation mode (disables gradient tracking)

all_preds = []
all_labels = []

with torch.no_grad(): # Turn off gradients to save memory and speed up computation
    for batch_X, batch_y in test_loader:
        probabilities = model(batch_X)
        # Convert probabilities to binary predictions (0 or 1) using a 0.5 threshold
        binary_preds = (probabilities >= 0.5).float()
        
        all_preds.extend(binary_preds.numpy())
        all_labels.extend(batch_y.numpy())

# Flatten arrays for scikit-learn
all_preds = np.array(all_preds).flatten()
all_labels = np.array(all_labels).flatten()

# Calculate and print metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f"Accuracy:  {accuracy:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("Confusion Matrix:\n", conf_matrix)

from fairlearn.metrics import (
    MetricFrame, 
    selection_rate, 
    false_positive_rate, 
    false_negative_rate,
    demographic_parity_difference,
    equalized_odds_difference
)

# ==========================================
# 8. Fairness Evaluation (Baseline)
# ==========================================
print("\n--- Step 5: Fairness Evaluation (Sex) ---")

# Define the specific metrics we want to track across groups
fairness_metrics = {
        "Accuracy": accuracy_score,
        "Selection Rate": selection_rate, # % predicted as >50K
        "False Positive Rate": false_positive_rate,
        "False Negative Rate": false_negative_rate,
}

# Create the MetricFrame to analyze metrics split by the 'sex' feature.
# Because test_loader had shuffle=False, all_preds aligns perfectly with A_test.
metric_frame_sex = MetricFrame(
    metrics=fairness_metrics,
    y_true=all_labels,
    y_pred=all_preds,
    sensitive_features=A_test['sex']
)

print("\nMetrics grouped by Sex:")
# We format this slightly so it prints neatly in the terminal
print(metric_frame_sex.by_group.to_string(float_format="%.4f"))

# Calculate overall fairness gaps
# Demographic Parity: Difference in Selection Rate between groups
dp_diff = demographic_parity_difference(
    y_true=all_labels, y_pred=all_preds, sensitive_features=A_test['sex']
)

# Equalized Odds: Maximum difference in FPR or TPR between groups
eo_diff = equalized_odds_difference(
    y_true=all_labels, y_pred=all_preds, sensitive_features=A_test['sex']
)

print(f"\nDemographic Parity Difference: {dp_diff:.4f}")
print(f"Equalized Odds Difference:     {eo_diff:.4f}")

from fairlearn.postprocessing import ThresholdOptimizer

# ==========================================
# 9. Bias Mitigation (Post-processing)
# ==========================================
print("\n--- Step 6: Bias Mitigation & Tradeoff Analysis ---")

# 1. Wrap the PyTorch model so Fairlearn can interact with it
class PyTorchWrapper:
    def __init__(self, model):
        self.model = model
        self.classes_ = np.array([0, 1]) # Required by Fairlearn
        
    def fit(self, X, y):
        return self # Model is already trained!
        
    def predict_proba(self, X):
        # Fairlearn needs the raw probabilities to calculate new thresholds
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            probs = self.model(X_tensor).numpy()
            return np.hstack([1-probs, probs]) # Returns [Prob of 0, Prob of 1]

wrapped_model = PyTorchWrapper(model)

# 2. Initialize the Threshold Optimizer
# We will optimize for Demographic Parity to close that 14.22% gap
optimizer = ThresholdOptimizer(
    estimator=wrapped_model,
    constraints="demographic_parity",
    predict_method='predict_proba',
    prefit=True 
)

# 3. Fit the optimizer to find the fair thresholds
# Note: In a strict production environment, you would use a separate validation 
# set here to prevent data leakage, but using train data is standard for this scope.
optimizer.fit(X_train_np, y_train_np, sensitive_features=A_train['sex'])

# 4. Generate the new, mitigated predictions on the test set
mitigated_preds = optimizer.predict(X_test_np, sensitive_features=A_test['sex'])

# 5. Evaluate the mitigated model
mitigated_accuracy = accuracy_score(all_labels, mitigated_preds)

metric_frame_mitigated = MetricFrame(
    metrics=fairness_metrics,
    y_true=all_labels,
    y_pred=mitigated_preds,
    sensitive_features=A_test['sex']
)

print("\nMitigated Metrics grouped by Sex:")
print(metric_frame_mitigated.by_group.to_string(float_format="%.4f"))

# 6. Document the Accuracy vs. Fairness Tradeoff
new_dp_diff = demographic_parity_difference(
    y_true=all_labels, y_pred=mitigated_preds, sensitive_features=A_test['sex']
)

print("\n--- The Tradeoff (Accuracy vs Fairness) ---")
print(f"Original Accuracy: {accuracy:.4f}  |  Mitigated Accuracy: {mitigated_accuracy:.4f}")
print(f"Original DP Diff:  {dp_diff:.4f}  |  Mitigated DP Diff:  {new_dp_diff:.4f}")