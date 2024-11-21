import torch
import torch.nn as nn
import torch.optim as optim
import json
import sys
#import grpc        my python is weird so im not 100% sure how to do these imports
#import modelservice_pb2
#import modelservice_pb2_grpc
from time import time, sleep
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, Dataset
import random
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(f"USING: {DEVICE}")


class JSONDataset(Dataset):
    """
    A PyTorch Dataset for loading data from a JSON file.
    """
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            self.data = json.load(file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        data = torch.tensor(sample["data"], dtype=torch.float32)  # Convert back to tensor
        label = torch.tensor(sample["label"], dtype=torch.long)  # Convert back to tensor
        return data, label



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)  # Binary classification output
        self.relu = nn.ReLU()
        # super(SimpleCNN, self).__init__()
        # self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(3136, 128)
        # self.fc2 = nn.Linear(128, 10)  # Binary classification output
        # self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        # x = self.sigmoid(self.fc2(x))  # Binary output
        x = self.fc2(x)  # Binary output
        
        return x.squeeze()
    



def train_model(dataloader, model, criterion, optimizer_fn=optim.Adam, epochs=5):
    
    model = model.to(DEVICE)
    optimizer = optimizer_fn(model.parameters())
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(inputs)
            oh_labels = nn.functional.one_hot(labels, 10).float()
            # print(inputs.shape, outputs.shape, labels.shape)
            
            loss = criterion(outputs, oh_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            # print(predicted.shape, labels.shape)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # print(f"\nEpoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}, Accuracy: {correct / total:.4f}")
        print(f"Train Accuracy: {correct / total:.4f}")
        # torch.save(model.state_dict(), f"model_epoch{epoch}.pth")

    return model

def evaluate(model, dataloader, criterion):
    """
    Evaluates the model on a test dataset.
    
    Parameters:
        model (torch.nn.Module): The trained model to evaluate.
        dataloader (DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function used for evaluation.

    Returns:
        dict: A dictionary containing the average loss and accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            # labels = labels.view(-1, 1)  # Reshape labels for binary classification
            oh_labels = nn.functional.one_hot(labels, 10).float()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, oh_labels)
            total_loss += loss.item()
            
            # Compute predictions and accurac
            
            # Track loss and accuracy
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            # print(predicted.shape, labels.shape)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return f"Validation Accuracy: {accuracy:.4f}"
    # return {"loss": avg_loss, "accuracy": accuracy}



def aggregate_models(model_paths, base_model_class):
    """
    Aggregates model weights from multiple .pth files using Federated Averaging (FedAvg).
    
    Parameters:
        model_paths (list): List of file paths to the .pth files (one per client).
        base_model_class (torch.nn.Module): Class of the base model (used to initialize the aggregated model).
        DEVICE (str): Device to load and process the models ("cpu" or "cuda").
    
    Returns:
        aggregated_model (torch.nn.Module): The model with aggregated weights.
    """
    
    # Load the state dictionaries from all models
    state_dicts = [torch.load(path, map_location=DEVICE, weights_only=False) for path in model_paths]
    
    # Initialize the base model and its state_dict
    base_model = base_model_class().to(DEVICE)
    aggregated_state_dict = base_model.state_dict()
    
    # Initialize the aggregated weights as zero
    for key in aggregated_state_dict.keys():
        aggregated_state_dict[key] = torch.zeros_like(aggregated_state_dict[key], dtype=torch.float32)
    
    # Aggregate weights from all models
    num_models = len(state_dicts)
    for state_dict in state_dicts:
        for key in aggregated_state_dict.keys():
            aggregated_state_dict[key] += state_dict[key]
    
    # Average the weights
    for key in aggregated_state_dict.keys():
        aggregated_state_dict[key] /= num_models
    
    # Load the aggregated weights into the base model
    base_model.load_state_dict(aggregated_state_dict)
    return base_model


def main():
    # lowkey could boot the go client at the start 
    # TODO this makes more sense with NLP tasks maybe
    
    
    if len(sys.argv) > 1:
        dataset = JSONDataset(sys.argv[1])
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    else:
        print("Usage: python prototype.py [data json]")
        exit(1)

    criterion = nn.CrossEntropyLoss()
    model = SimpleCNN()
    
    # torch.save(model.state_dict(), f"my_model.pth")
    start = time() # agg every 30 seconds
    
    test_dataset = JSONDataset("test_data.json")
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    while True:
        
        end = time()
        
        if end - start > 15:
            # this would be an RPC
            print("BLOCK and requesting other models to be pushed into the to_aggregate folder")
            sleep(3)

            """
            so what should go here is a call to the CollectModels funciton of the go client
            this passes:
            - secret key (set at boot of go client via flag)  
            - number of models to collect
            the go client then response with the actual number of models that have been collected
                NOTE: could be lower than requested due to lack of peers or connection failures
            ideally should be some timeout where if there are no models recieved by then it just
            goes back to training maybe. idk though
            """


            print("models received")
            
            agg_paths = [os.path.join("to_aggregate", x) for x in os.listdir("to_aggregate")]
            agg_paths = random.sample(agg_paths, 3) # this is only bc we are not actaully sending models yet -- TODO remove eventually
            model = aggregate_models(agg_paths, SimpleCNN)
            
            start = time()
            
        else:
            model = train_model(dataloader, model, criterion, epochs=1)
        
        print(evaluate(model, test_dataloader, criterion))
        torch.save(model.state_dict(), f"my_model.pth")
        

if __name__ == "__main__":
    main()