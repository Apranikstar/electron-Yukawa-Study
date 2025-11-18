import warnings
import uproot
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

def process_single_file(root_file, output_dir, tree_name, max_events):
    """Process a single ROOT file and save as pickle"""
    try:
        with uproot.open(root_file) as f:
            if tree_name not in f:
                return f"[!] No tree named '{tree_name}' in {root_file.name}"
            
            tree = f[tree_name]
            n_entries = tree.num_entries
            n_to_read = min(n_entries, max_events)
            
            # Read data more efficiently
            df = tree.arrays(
                library="pd", 
                entry_stop=n_to_read
            )
            
            # Save with compression for smaller files and faster I/O
            output_file = output_dir / (root_file.stem + ".pkl")
            df.to_pickle(output_file, compression="gzip", protocol=5)
            
            return f"✓ {root_file.name}: {len(df):,} events → {output_file.name}"
            
    except Exception as e:
        return f"✗ {root_file.name}: {e}"

def main():
    print("Starting parallel ROOT to pickle conversion...")
    
    input_dir = Path("/eos/experiment/fcc/ee/analyses/case-studies/higgs/electron_yukawa/DataGen/on-shell-electron/")
    output_dir = Path(".")
    tree_name = "events"
    max_events = 300_000
    
    # Get all ROOT files
    root_files = list(input_dir.glob("*.root"))
    n_files = len(root_files)
    
    if n_files == 0:
        print("No ROOT files found!")
        return
    
    print(f"Found {n_files} ROOT file(s)")
    
    # Determine optimal number of workers (don't use all cores to avoid system slowdown)
    n_workers = max(1, mp.cpu_count() - 2)
    print(f"Using {n_workers} parallel workers\n")
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_file, rf, output_dir, tree_name, max_events): rf
            for rf in root_files
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"[{i}/{n_files}] {result}")
    
    print("\n✓ All files processed!")

if __name__ == "__main__":
    main()
