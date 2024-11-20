import json
import torch
import sys
import os
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import datasets, transforms

def generate_partition_data(batch_size=32, num_clients=10, labels_per_client=7, train_proportion=0.8):
    # Step 1: Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    
    train_size = int(train_proportion * len(mnist_dataset))  # 80% training data
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
    
    


def save_dataloader_to_json(dataloader, filename):
    """
    Saves the data and labels from a DataLoader to a JSON file.

    Parameters:
        dataloader (DataLoader): The DataLoader to save.
        filename (str): The JSON file to save the data to.
    """
    data_list = []

    for inputs, labels in dataloader:
        # Convert tensors to lists for JSON serialization
        inputs = inputs.numpy().tolist()
        labels = labels.numpy().tolist()
        
        # Append each sample and its label as a dictionary
        for input_sample, label in zip(inputs, labels):
            data_list.append({"data": input_sample, "label": label})
    
    # Save to JSON file
    with open(filename, 'w+') as json_file:
        json.dump(data_list, json_file, indent=4)
    print(f"Data saved to {filename}")


def main():
    
    loaders, test = generate_partition_data()
    os.mkdir("client_data")
    
    for k, v in loaders.items():
        save_dataloader_to_json(v, f"client_data/{k}.json")
    
    save_dataloader_to_json(test, "test_data.json")
    


if __name__ == "__main__":
    main()