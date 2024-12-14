import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the output files
output_dir = "./outputs"  # Update this to your actual directory path

# List to hold data from all files
all_data = []

# Read all output files
for filename in os.listdir(output_dir):
    if filename.endswith(".txt"):  # Assuming the files are .txt
        filepath = os.path.join(output_dir, filename)
        data = pd.read_csv(filepath, header=None, names=["Accuracy", "Time"])
        data["File"] = filename  # Add a column to identify the file
        data["Line Number"] = range(1, len(data) + 1)  # Add line numbers (1-indexed)
        all_data.append(data)

# Combine all data into a single DataFrame
df = pd.concat(all_data, ignore_index=True)

# Find the minimum time across all files
min_time = df["Time"].min()

# Adjust times to start at 0
df["Adjusted Time"] = df["Time"] - min_time

# Plot the number of lines processed versus adjusted time
plt.figure(figsize=(12, 8))

for i, (filename, group) in enumerate(df.groupby("File")):
    plt.plot(group["Adjusted Time"], group["Line Number"], label=f"Peer {i} Fit")

# Customize the plot
plt.title("Number of Training Cycles Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Number of Training Cycles")
plt.legend()
plt.grid(True)
plt.show()
