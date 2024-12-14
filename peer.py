import torch
import torch.nn as nn
import torch.optim as optim
import json
import sys
import grpc
import proto.modelservice_pb2 as pb2
import proto.modelservice_pb2_grpc as pb2_grpc
from time import time, sleep
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, Dataset
import random
import fcntl
import shutil
import importlib
import inspect

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(f"USING: {DEVICE}")

def get_model_arch(module_name, filepath=None): #filepath defaults to module_name.py
    # Dynamically import the module
    
    if filepath is None:
        filepath = module_name + ".py"
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the single class in the module
    classes = [obj for name, obj in inspect.getmembers(module, inspect.isclass) 
               if obj.__module__ == module_name]

    if len(classes) != 1:
        raise ValueError("The module must contain exactly one class.")
    
    return classes[0]  # Return the single class

# print(get_model_arch("model_arch"))

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

    return accuracy
    # return {"loss": avg_loss, "accuracy": accuracy}

def bad_file_completion(path, timeout=5):
    
    prev = -1
    start_time = time()
    
    while time() - start_time < timeout:
        curr = os.path.getsize(path)
        
        if curr == prev:
            return True
        
        prev = curr
        sleep(1)
        
    return False

def aggregate_models(data_dir, agg_dir, base_model_class, timeout=5):
    """
    Aggregates model weights from multiple .pth files using Federated Averaging (FedAvg).
    
    Parameters:
        model_paths (list): List of file paths to the .pth files (one per client).
        base_model_class (torch.nn.Module): Class of the base model (used to initialize the aggregated model).
        DEVICE (str): Device to load and process the models ("cpu" or "cuda").
    
    Returns:
        aggregated_model (torch.nn.Module): The model with aggregated weights.
    """
    
    if len(os.listdir(agg_dir)) == 0:
        return None
    stime = time()
    # Load the state dictionaries from all models

    while time() - stime < timeout:
        if os.path.exists(os.path.join(data_dir, ".DONE")):
            
            try:
                agg_paths = [os.path.join(agg_dir, x) for x in os.listdir(agg_dir)]

                state_dicts = [torch.load(path, map_location=DEVICE, weights_only=False) for path in agg_paths]
                
                # state_dicts = []
                
                # # TODO maybe?
                # for path in model_paths:
                    
                #     fp = open(path, 'r')
                #     fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Attempt to acquire an exclusive lock
                #     fcntl.flock(fp, fcntl.LOCK_UN)  # Release the lock if successful
                #     state_dicts.append(torch.load(path, map_location=DEVICE, weights_only=False))
                
                
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
                os.remove(os.path.join(data_dir, ".DONE"))
                return base_model
            
            except:
                return None
        sleep(1)
    print("TIMEOUT")
    return None

def collect_models(client, secret_key, num_models, timeout=10):
    """
    Calls the CollectModels gRPC function on the Go server.

    Args:
        client: The gRPC client object.
        secret_key (str): The secret key for authentication.
        num_models (int): The number of models to collect.
        timeout (int): Timeout in seconds for the gRPC call.

    Returns:
        int: The number of models actually collected.
    """
    request = pb2.CollectModelsRequest(key=secret_key, num=num_models)

    try:
        response = client.CollectModels(request, timeout=timeout)
        if response.success:
            print(f"Successfully collected?: {response.success}.")
            return response.success
        else:
            print("Failed to collect models. Unauthorized or other issue.")
            return 0
    except grpc.RpcError as e:
        print(f"gRPC error: {e.code()} - {e.details()}")
        return 0

def main():
    # lowkey could boot the go client at the start 
    # TODO this makes more sense with NLP tasks maybe
    
    if len(sys.argv) > 2:
        dataset = JSONDataset(sys.argv[1])
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        port = int(sys.argv[2])
        data_dir = f"{port}_data"
        agg_dir = os.path.join(data_dir, "agg")
        os.mkdir(agg_dir) 
    else:
        print("Usage: python prototype.py [data json] [port]")
        exit(1)
    
    # bood timeout
    stime = time()
    timeout = 15

    while time() - stime < timeout:
        if os.path.exists(os.path.join(data_dir, ".BOOT_DONE")):
            os.remove(os.path.join(data_dir, ".BOOT_DONE"))
            break
        sleep(1)
        
    if time() - stime >= timeout:
        print("BOOT TIMEOUT")
        sys.exit(12)
    

    criterion = nn.CrossEntropyLoss()
    
    # model = model = torch.load("model.pth")
    # model = SimpleCNN() # TODO this should be loaded
    model_class = get_model_arch("model_arch", os.path.join(data_dir, "model_arch.py"))
    
    model = model_class()
    
    # print(model)
    # print(SimpleCNN)
    # print(temp)
    # torch.save(model.state_dict(), f"my_model.pth")
    start = time() # agg every 30 seconds
    
    test_dataset = JSONDataset("test_data.json")
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    while True:
        
        end = time()
        
        if end - start > 15:
            # this would be an RPC
            # print("BLOCK and requesting other models to be pushed into the to_aggregate folder")
            # sleep(3)
            
            server_address = f"localhost:{port}"

            # Connect to the server
            with grpc.insecure_channel(server_address) as channel:
                client = pb2_grpc.ModelServiceStub(channel)

                # Secret key and number of models to collect
                secret_key = "secret"

                # Call the function
                collect_models(client, secret_key, num_models=1, timeout=10)
                # print(f"Models collected?: {models_collected}")

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

            

            # print("models received")
            
            # agg_paths = random.sample(agg_paths, 3) # this is only bc we are not actaully sending models yet -- TODO remove eventually
            agg = aggregate_models(data_dir, agg_dir, model_class)
            
            if agg is not None:
                model = agg
            
            start = time()
            
        else:
            model = train_model(dataloader, model, criterion, epochs=1)
        
        accuracy = evaluate(model, test_dataloader, criterion)
        
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        with open(f"outputs/{port}_output.txt", "a+") as file:
            file.write(f"{accuracy}, {time()}\n")

        temp_path = os.path.join(data_dir, "my_model.tmp")
        try: 
            torch.save(model.state_dict(), temp_path)
            
            os.replace(temp_path, os.path.join(data_dir, "my_model.pth"))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)        

if __name__ == "__main__":
    main()
    

