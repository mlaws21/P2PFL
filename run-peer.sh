#!/bin/bash

# Check if a port argument is provided
if [ -z "$1" ]; then
  echo "Error: Missing 'num' argument."
  echo "Usage: $0 <NUM>"
  exit 1
fi

# if [ -z "$2" ]; then
#   echo "Error: Missing 'data' argument."
#   echo "Usage: $0 <port> <data>"
#   exit 1
# fi

# Validate that the port is a valid number
if ! [[ "$1" =~ ^[0-9]+$ ]]; then
  echo "Error: Num must be a numeric value."
  exit 1
fi

# Assign the port argument
NUM=$1


# Output the provided port

GOCMD="go run modelservice_server/modelservice_server.go -port 800$NUM -local_model_path ./my_model$NUM.pth -collected_models_path ./agg"
PYCMD="python peer.py ./client_data/$NUM.json 800$NUM"

eval $GOCMD &
eval $PYCMD

