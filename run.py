# python run.py 8000 localhost:8000 model_arch.py
# python run.py 8001 localhost:8000 model_arch.py
# python run.py 8002 localhost:8001 model_arch.py


import sys
import subprocess
import re

def main():
    # Check if a port argument is provided
    if len(sys.argv) < 3:
        print("Error: Missing arguments.")
        print("Usage: python run.py <PORT> <BOOT_IP> <MODEL_ARCH>")
        sys.exit(1)

    # Assign the port argument
    port_arg = sys.argv[1]
    boot_arg = sys.argv[2]
    
    if len(sys.argv) > 3:
        arch_arg = sys.argv[3]
    else:
        arch_arg = ""
    

    # Validate that the port is a valid number
    if not re.match(r'^\d+$', port_arg):
        print("Error: PORT must be a numeric value.")
        sys.exit(1)

    # Format the port and paths
    port_number = port_arg#f"800{port_arg}"
    local_model_path = f"./{port_number}_data/my_model.pth"
    collected_models_path = f"./{port_number}_data/agg"
    client_data_path = f"./client_data/{port_arg}.json"

    # Commands
    go_cmd = [
        "go", "run", "modelservice_server/modelservice_server.go",
        "-port", port_number,
        "-boo_ip", boot_arg,
        "-local_model_path", local_model_path,
        "-arch", arch_arg
        # "-collected_models_path", collected_models_path
    ]
    py_cmd = ["python", "peer.py", client_data_path, port_number]

    try:
        # Run the commands
        go_process = subprocess.Popen(go_cmd)
        py_process = subprocess.Popen(py_cmd)

        # Wait for both processes to complete
        go_process.wait()
        py_process.wait()

        # Check exit statuses
        if go_process.returncode != 0:
            print(f"Go command failed with exit code {go_process.returncode}")
        if py_process.returncode != 0:
            print(f"Python command failed with exit code {py_process.returncode}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Script execution complete.")

if __name__ == "__main__":
    main()
