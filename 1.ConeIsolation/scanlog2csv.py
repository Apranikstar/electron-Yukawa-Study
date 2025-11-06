import re
import pandas as pd

logfile = "simulation.log"  # ✅ Change to your filename

# Regex patterns
sim_header_re = re.compile(r"Running: drmin=([0-9.]+), drmax=([0-9.]+), selection=([0-9.]+)")
process_re = re.compile(r"\.\/output\/([^.]+)\.root")
reduction_re = re.compile(r"Reduction factor local:\s+([0-9.]+)")
captured_re = re.compile(r"Captured Output file:\s+\.\/output\/([^.]+)\.root, Reduction factor:\s+([0-9.]+)")
events_processed_re = re.compile(r"Total events processed:\s+([\d,]+)")
result_events_re = re.compile(r"No\. result events:\s+([\d,]+)")
events_per_sec_re = re.compile(r"Events processed\/second:\s+([0-9,]+)")
elapsed_re = re.compile(r"Elapsed time.*:\s+(\d+:\d+:\d+)")

results = []

current_sim = {}
current_process = None
process_data = {}

def store_process():
    """Store results for the current process if all necessary information exists"""
    if current_sim and current_process and "reduction_factor" in process_data:
        results.append({
            "experiment_name": current_sim["name"],
            "drmin": current_sim["drmin"],
            "drmax": current_sim["drmax"],
            "selection": current_sim["selection"],
            "process": current_process,
            "reduction_factor": process_data.get("reduction_factor"),
            "events_processed": process_data.get("events_processed"),
            "result_events": process_data.get("result_events"),
            "events_per_second": process_data.get("events_per_sec"),
            "elapsed_time": process_data.get("elapsed_time"),
        })

with open(logfile, "r") as f:
    for line in f:

        # Detect simulation conditions
        sim = sim_header_re.search(line)
        if sim:
            # Store previous process before starting new sim
            store_process()
            process_data = {}
            current_process = None

            drmin, drmax, selection = sim.groups()
            current_sim = {
                "name": f"dr{drmin}_dr{drmax}_sel{selection}",
                "drmin": float(drmin),
                "drmax": float(drmax),
                "selection": float(selection)
            }
            continue

        # Detect process name (any heuristic .root in output/)
        proc = process_re.search(line)
        if proc:
            # If previous process exists, save it before switching
            store_process()
            current_process = proc.group(1)
            process_data = {}
            continue

        # Detect captured summary
        cap = captured_re.search(line)
        if cap:
            current_process = cap.group(1)
            process_data["reduction_factor"] = float(cap.group(2))
            store_process()
            current_process = None
            process_data = {}
            continue

        # Reduction factor local
        red = reduction_re.search(line)
        if red:
            process_data["reduction_factor"] = float(red.group(1))
            continue

        # Performance and event stats
        ep = events_processed_re.search(line)
        if ep:
            process_data["events_processed"] = int(ep.group(1).replace(",", ""))

        re_event = result_events_re.search(line)
        if re_event:
            process_data["result_events"] = int(re_event.group(1).replace(",", ""))

        evps = events_per_sec_re.search(line)
        if evps:
            process_data["events_per_sec"] = int(evps.group(1).replace(",", ""))

        el = elapsed_re.search(line)
        if el:
            process_data["elapsed_time"] = el.group(1)

# Final commit flush
store_process()

df = pd.DataFrame(results)
df.to_csv("parsed_results.csv", index=False)

print("✅ Parsing Complete!")
print(f"✅ Extracted {len(df)} entries")
print("📄 Saved to: parsed_results.csv")
print(df.head())
