#!/bin/bash

# Check if a range of ports is provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <start_port> <end_port>"
    exit 1
fi

start_port=$1
end_port=$2

# Validate that the inputs are numbers
if ! [[ $start_port =~ ^[0-9]+$ ]] || ! [[ $end_port =~ ^[0-9]+$ ]]; then
    echo "Error: Ports must be numeric."
    exit 1
fi

# Iterate over the port range
for port in $(seq $start_port $end_port); do
    # Find the PID of the process using the port
    pid=$(lsof -ti :$port)

    if [ -n "$pid" ]; then
        echo "Killing process $pid on port $port"
        kill $pid
    else
        echo "No process found on port $port"
    fi
done

echo "Completed killing processes in the port range $start_port-$end_port."
