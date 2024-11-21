#!/bin/bash

generate_proto() {
    echo "Generating protobuf code..."
    protoc --go_out=. \
           --go_opt=paths=source_relative \
           --go-grpc_out=. \
           --go-grpc_opt=paths=source_relative \
           proto/*.proto
    
    # lowkey should not have two differnt commands here but idk man
    python3 -m grpc_tools.protoc \
            -I. \
            --python_out=. \
            --grpc_python_out=. \
            proto/*.proto
}

generate_proto
