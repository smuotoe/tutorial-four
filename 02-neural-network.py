# Dataset: Bank Note Authentication
# Learning Objectives:
# 1. Understand basic neural network architecture
# 2. Implement forward pass in PyTorch
# 3. Train a model using PyTorch's training loop
# 4. Compare different network architectures

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

"""
The Bank Note Authentication dataset contains features extracted from images 
of genuine and forged banknotes. Features include variance, skewness, 
kurtosis, and entropy of the wavelet-transformed image.
"""


def load_and_preprocess_data():
    """
    Load and preprocess the banknote dataset.
    Returns preprocessed data as PyTorch tensors, ready for training.
    """
    # Load data
    col_names = ["variance", "skewness", "kurtosis", "entropy", "label"]
    features = col_names[:-1]
    bank_note_df = pd.read_csv(
        "data_banknote_authentication.txt", header=None, names=col_names
    )

    # Split features and target
    X = bank_note_df[features]
    y = bank_note_df["label"]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.LongTensor(y_train.values)
    y_test_tensor = torch.LongTensor(y_test.values)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


class OneLayerNet(nn.Module):
    """
    TODO: Implement a single-layer neural network
    1. Initialize the layer in __init__
    2. Define the forward pass
    """

    def __init__(self, input_size):
        super().__init__()
        # TODO: Create a single linear layer with input_size inputs and 2 outputs
        # Hint: Use nn.Linear
        self.fc1 = None

    def forward(self, x):
        # TODO: Implement the forward pass
        # Hint: Simply apply the linear layer
        return None


class TwoLayerNet(nn.Module):
    """
    TODO: Implement a two-layer neural network
    1. Initialize the layers in __init__
    2. Define the forward pass with ReLU activation
    """

    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        # TODO: Create two linear layers and a ReLU activation
        # Hint: First layer maps input_size to hidden_size
        # Hint: Second layer maps hidden_size to 2 (number of classes)
        self.fc1 = None
        self.relu = None
        self.fc2 = None

    def forward(self, x):
        # TODO: Implement the forward pass
        # Hint: x -> fc1 -> relu -> fc2
        return None


def train_model(model, X_train, y_train, num_epochs, criterion, optimizer):
    """
    TODO: Implement the training loop
    1. Forward pass
    2. Compute loss
    3. Backward pass
    4. Update weights
    5. Track and print metrics
    """
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()

        # TODO: Implement forward pass
        outputs = None

        # TODO: Compute loss
        loss = None

        # TODO: Implement backward pass and optimization
        # Hint: Remember to zero gradients first
        optimizer.zero_grad()
        # Your code here

        # Calculate accuracy
        _, y_pred = torch.max(outputs.data, 1)
        accuracy = (y_pred == y_train).sum().item() / len(y_train)

        # if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Loss: {loss.item():.4f}, "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

    return model


def evaluate_model(model, X_test, y_test):
    """
    TODO: Implement model evaluation
    1. Set model to evaluation mode
    2. Compute predictions
    3. Calculate accuracy
    """
    model.eval()
    with torch.no_grad():
        # TODO: Compute test predictions and accuracy
        # Your code here
        pass


# Set random seeds for reproducibility
def set_seed(seed=42):
    """
    Set seeds for all sources of randomness to ensure reproducible results.
    """
    torch.manual_seed(seed)  # PyTorch random number generator
    torch.cuda.manual_seed(seed)  # CUDA random number generator
    torch.cuda.manual_seed_all(seed)  # CUDA random number generator for all GPUs
    torch.backends.cudnn.deterministic = True  # Make CUDA deterministic
    torch.backends.cudnn.benchmark = False  # Disable CUDA benchmarking


def main():
    # Set random seed
    set_seed(42)

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    input_size = 4  # Number of features

    # TODO: Create model instances
    # Experiment with both architectures!
    one_layer_model = OneLayerNet(input_size=input_size)
    two_layer_model = TwoLayerNet(input_size=input_size)

    # TODO: Define loss function and optimizer
    # Hint: Use CrossEntropyLoss and SGD with momentum
    criterion = None
    optimizer = None

    # Train and evaluate models
    print("Training One-Layer Network:")
    trained_model = train_model(
        model=one_layer_model,
        X_train=X_train,
        y_train=y_train,
        num_epochs=50,
        criterion=criterion,
        optimizer=optimizer
    )

    print("\nEvaluating One-Layer Network:")
    evaluate_model(trained_model, X_test, y_test)

    # TODO: Repeat training and evaluation with two-layer network
    # Compare the results!


if __name__ == "__main__":
    main()

"""
Exercise Tasks:
1. Complete all TODO sections in the code
2. Train both network architectures
3. Compare their performance
4. Try modifying hyperparameters:
   - Learning rate
   - Momentum
   - Number of epochs
   - Hidden layer size (for TwoLayerNet)

Questions to Consider:
1. Which model performs better? Why?
2. How does the hidden layer size affect performance?
3. What happens if you change the learning rate?
4. Could you improve the model further? How?
"""