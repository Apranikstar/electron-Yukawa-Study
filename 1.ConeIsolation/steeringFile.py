import json
import subprocess
import numpy as np
import re
import csv

# -----------------------------
# Parameters
# -----------------------------
drmin = [0.01]
drmax = [0.2, 0.4, 0.6, 0.8, 1.0]
num_selection = 20

# Prepare list of args: [drmin, drmax, selection_array]
args = []
for dmax in drmax:
    selections = np.linspace(drmin[0], dmax, num_selection)
    args.append([drmin[0], dmax, selections])

# Storage for results
results = []

# -----------------------------
# Run parameter scan
# -----------------------------
for drmin_val, drmax_val, selections in args:
    for selection in selections:
        # Write config.json
        cfg = {
            "drmin": float(drmin_val),
            "drmax": float(drmax_val),
            "selection": float(selection)
        }
        with open("config.json", "w") as f:
            json.dump(cfg, f)

        # Run fccanalysis
        cmd = ["fccanalysis", "run", "cone.py", "--ncpus", "64"]
        print(f"\nRunning: drmin={drmin_val}, drmax={drmax_val:.4f}, selection={selection:.4f}")

        # Stream output and capture
        stdout_lines = []
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            for line in proc.stdout:
                print(line, end="")  # live output
                stdout_lines.append(line)
            proc.wait()

        stdout = "".join(stdout_lines)

        print(f"  -> Captured Output file: {output_file}, Reduction factor: {reduction_factor:.6f}")

      )

print("Scanner the ranges successfuly!")
