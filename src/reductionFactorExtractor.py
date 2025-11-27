import re
import csv

# Path to your log file
log_file = "isoEl1isoMu0.log"  # replace with your actual .log file
# Output CSV file
csv_file = "isoEl1isoMu0.csv"

results = []

with open(log_file, "r") as f:
    lines = f.readlines()

process_name = None

for i, line in enumerate(lines):
    # Detect the output file line
    if "Output file path:" in line:
        if i + 1 < len(lines):
            match = re.search(r"./output/isoel/(.+)\.root", lines[i + 1].strip())
            if match:
                process_name = match.group(1)

    # Detect the SUMMARY section
    if "SUMMARY" in line:
        # Look for the reduction factor in the next few lines
        for j in range(i+1, min(i+10, len(lines))):
            red_match = re.search(r"Reduction factor local:\s+([\d.]+)", lines[j])
            if red_match and process_name:
                reduction_factor = float(red_match.group(1))
                results.append((process_name, reduction_factor))
                process_name = None  # reset for next block
                break

# Save results to CSV
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Process", "Reduction Factor"])
    writer.writerows(results)

print(f"Results saved to {csv_file}")
