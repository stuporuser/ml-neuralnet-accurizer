import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from itertools import product
import optuna
from sklearn.model_selection import ParameterGrid
import pandas as pd

# (1) Load and preprocess MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# (2) Define Neural Network Model with varying number of layers, dropout, and activation functions
class NeuralNet(nn.Module):
    def __init__(self, activation_fn, dropout_rate, num_layers):
        super(NeuralNet, self).__init__()

        # Create layers list
        layers = []

        # First layer (input to first hidden layer)
        layers.append(nn.Linear(28 * 28, 128))  # Assuming input image size of 28x28

        # Add hidden layers with dropout and activation
        for _ in range(num_layers - 1):  # Add additional hidden layers
            layers.append(nn.Linear(128, 128))  # You can adjust this for more complexity
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(activation_fn())  # Instantiate the activation function here
        
        # Output layer (10 classes for MNIST)
        layers.append(nn.Linear(128, 10))  # 10 output classes

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten image
        return self.model(x)  # Pass through layers

# (3) Initialize Model, Loss, and Optimizer
def run_experiment(batch_size, learning_rate, dropout_rate, epochs, activation_fn, num_layers, optimizer_type):
    model = NeuralNet(activation_fn, dropout_rate, num_layers)  # Create model

    # Select optimizer
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    criterion = nn.CrossEntropyLoss()  # Softmax + CrossEntropy

    # Train the Neural Network
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()  # Reset gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            running_loss += loss.item()
        
        # Print loss every epoch
        #print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}")
    
    # Test the Model
    correct = 0
    total = 0
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    #print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# (4) Hyperparameter search grid
batch_sizes = [32, 64] #[32, 64] #, 128]  # Different batch sizes
learning_rates = [0.001] #[0.0005, 0.001] #, 0.01]  # Different learning rates
dropout_rates = [0.0, 0.2] #, 0.5]  # Different dropout rates
activations = [nn.ReLU, nn.LeakyReLU, nn.Sigmoid]  # Different activations (passed as classes)
optimizers = ['adam', 'rmsprop'] #, 'sgd']  # Different optimizers
epochs = [5] #[20, 50, 100]  # Different epochs
num_layers = [2, 3] #[1, 2, 3, 4]  # Different numbers of layers (depth of the network)

# (5) Grid search using sklearn.model_selection.ParameterGrid
param_grid = {
    'batch_size': batch_sizes,
    'learning_rate': learning_rates,
    'dropout_rate': dropout_rates,
    'activation_fn': activations,
    'optimizer_type': optimizers,
    'epochs': epochs,
    'num_layers': num_layers
}

# (6) Run grid search
#results = []

## Loop over all combinations of hyperparameters
#for params in ParameterGrid(param_grid):
#    print(f"Running: {params}...", end="", flush=True)
#    #run_experiment(batch_size, learning_rate, dropout_rate, epochs, activation_fn, num_layers, optimizer_type):
#    accuracy = run_experiment(params['batch_size'], params['learning_rate'], params['dropout_rate'], params['epochs'],
#                              params['activation_fn'], params['num_layers'], params['optimizer_type'])
#    results.append({
#        "batch_size": params['batch_size'],
#        "learning_rate": params['learning_rate'],
#        "dropout_rate": params['dropout_rate'],
#        "activation_fn": str(params['activation_fn']),
#        "optimizer_type": params['optimizer_type'],
#        "epochs": params['epochs'],
#        "num_layers": params['num_layers'],
#        "accuracy": accuracy
#    })
#
## (7) Save and display results
#df = pd.DataFrame(results)
#df.to_csv("experiment_results_grid_search_with_depth.csv", index=False)
#
#print("\nGrid Search Experiment Results:")
#print(df)

# (8) Random search with Optuna
def objective(trial):
    # Hyperparameters repeated in this scope for debugging simplicity.
    batch_sizes = [32, 64] #[32, 64] #, 128]  # Different batch sizes
    learning_rates = [0.001] #[0.0005, 0.001] #, 0.01]  # Different learning rates
    dropout_rates = [0.0, 0.2] #, 0.5]  # Different dropout rates
    activations = [nn.ReLU, nn.LeakyReLU, nn.Sigmoid]  # Different activations (passed as classes)
    optimizers = ['adam', 'rmsprop'] #, 'sgd']  # Different optimizers
    epochs = [5] #[20, 50, 100]  # Different epochs
    num_layers = [2, 3] #[1, 2, 3, 4]  # Different numbers of layers (depth of the network)

    batch_size = trial.suggest_categorical('batch_size', batch_sizes)
    learning_rate = trial.suggest_float('learning_rate', 0.0005, 0.01, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    activation_fn = trial.suggest_categorical('activation_fn', [nn.ReLU, nn.LeakyReLU, nn.Sigmoid])  # Pass classes
    optimizer_type = trial.suggest_categorical('optimizer_type', optimizers)
    epochs = trial.suggest_int('epochs', 2, 5) #20, 100)
    num_layers = trial.suggest_categorical('num_layers', num_layers)

    accuracy = run_experiment(batch_size, learning_rate, dropout_rate, epochs, activation_fn, num_layers, optimizer_type)
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # Run 50 random trials

print("\nBest Trial:")
print(study.best_trial.params)
print("Best Accuracy:", study.best_trial.value)
