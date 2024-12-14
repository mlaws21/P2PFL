# P2PFL: Peer to Peer Federated Learning
Matt Laws and Nathan Vosburg

# Setup:
## Go Protobuf Build:
- go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
- go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
- export PATH=\$PATH:\$(go env GOPATH)/bin (in .rc file and source)
- ./build.sh

## Conda Build:
- conda env create -f environment.yaml (this is probably more than you need but it works)
- conda activate ml

## Test Go Clients:
- Example:
    - go run modelservice_server/modelservice_server.go -port 8001 -local_model_path ./my_model.pth -collected_models_path ./to_aggregate
        - this runs on localhost because no "boot_ip" flag was specified


# Running

- python main.py (this runs our UI)
