import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Subset
from tqdm import tqdm


def generate_partition_data(batch_size=32, num_clients=10, labels_per_client=7, train_proportion=0.8):
    # Step 1: Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    
    train_size = int(0.8 * len(mnist_dataset))  # 80% training data
    test_size = len(mnist_dataset) - train_size 


    train_dataset, test_dataset = random_split(mnist_dataset, [train_size, test_size])
    # Step 2: Group indices by labels
    label_to_indices = {i: [] for i in range(10)}  # Dictionary to hold indices for each label

    for idx, (_, label) in enumerate(train_dataset):
        # print(label)
        label_to_indices[label].append(idx)

    # Step 3: Partition the dataset for each client
    # Example: Assign labels 0 and 1 to client 1, labels 2 and 3 to client 2, etc.
    client_partitions = {}


    for client_id in range(num_clients):
        client_labels =  [x % 10 for x in range(client_id * labels_per_client, (client_id + 1) * labels_per_client)]
        client_indices = []
        for label in client_labels:
            client_indices.extend(label_to_indices[label])
        client_partitions[client_id] = client_indices

    # Step 4: Create DataLoaders for each client
    client_dataloaders = {}

    for client_id, indices in client_partitions.items():
        client_dataset = Subset(train_dataset, indices)
        client_dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
        client_dataloaders[client_id] = client_dataloader

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return client_dataloaders, test_loader
    # # Verify: Print sizes of datasets for each client
    # for client_id, dataloader in client_dataloaders.items():
    #     print(f"Client {client_id}: {len(dataloader.dataset)}")
        
    #     all_labels = []
    #     for data, labels in dataloader:
    #         all_labels.extend(labels.numpy())  # Convert tensor to numpy array for easier processing

    #     # Step 3: Convert the list of labels to a tensor or numpy array if needed
    #     all_labels = torch.tensor(all_labels)
    #     print(f"Unique labels: {torch.unique(all_labels)}")


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)  # Binary classification output
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary output

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        # x = self.sigmoid(self.fc2(x))  # Binary output
        x = self.fc2(x)  # Binary output
        
        return x.squeeze()
    



def train_model(dataloader, model, criterion, optimizer_fn=optim.Adam, epochs=5):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING: {device}")
    model = model.to(device)
    optimizer = optimizer_fn(model.parameters())
    
    model.train()
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(dataloader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

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

        print(f"\nEpoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}, Accuracy: {correct / total:.4f}")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
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
    return {"loss": avg_loss, "accuracy": accuracy}


dataloaders, test_dataloader = generate_partition_data()

criterion = nn.CrossEntropyLoss()



model = SimpleCNN()
# model.load_state_dict(torch.load("model.pth"))

# print(evaluate(model, test_dataloader, criterion))


for i, ele in dataloaders.items():
    model = train_model(ele, model, criterion)
    torch.save(model.state_dict(), f"model{i}.pth")
    


