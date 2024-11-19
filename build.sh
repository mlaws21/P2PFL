#!/bin/bash

generate_proto() {
    echo "Generating protobuf code..."
    protoc --go_out=. \
           --go_opt=paths=source_relative \
           --go-grpc_out=. \
           --go-grpc_opt=paths=source_relative \
           proto/*.proto
}

# Run generation
generate_proto
