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
        all_data.append(data)

# Combine all data into a single DataFrame
df = pd.concat(all_data, ignore_index=True)

# Find the minimum time across all files
min_time = df["Time"].min()

# Adjust times to start at 0
df["Adjusted Time"] = df["Time"] - min_time

# Plot each file's data with fit lines
plt.figure(figsize=(12, 8))

# Group by file and plot individual data and fits
for i, (filename, group) in enumerate(df.groupby("File")):
    # Plot the data points
    plt.scatter(group["Adjusted Time"], group["Accuracy"])

    # Fit a line to the data
    coeffs = np.polyfit(group["Adjusted Time"], group["Accuracy"], 1)  # Linear fit
    fit_line = np.polyval(coeffs, group["Adjusted Time"])
    plt.plot(group["Adjusted Time"], fit_line, linestyle="--", label=f"Client {i} Fit")

# Fit a single line across all data points (combined)
global_coeffs = np.polyfit(df["Adjusted Time"], df["Accuracy"], 1)
global_fit_line = np.polyval(global_coeffs, df["Adjusted Time"])

# Plot the global fit line
plt.plot(df["Adjusted Time"], global_fit_line, "r-", linewidth=3, label="Average Fit")

# Customize the plot
plt.title("Accuracy vs. Time of Clients")
plt.xlabel("Time (seconds)")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.show()
