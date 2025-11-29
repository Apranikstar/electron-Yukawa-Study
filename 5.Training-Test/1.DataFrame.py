import warnings
import uproot
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

def process_single_file(root_file, output_dir, tree_name, max_events, use_compression):
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
            
            # Save with optional compression
            output_file = output_dir / (root_file.stem + ".pkl")
            if use_compression:
                df.to_pickle(output_file, compression="gzip", protocol=5)
                comp_status = "(compressed)"
            else:
                df.to_pickle(output_file, protocol=5)
                comp_status = "(uncompressed)"
            
            # Get file size for reporting
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            
            return f"✓ {root_file.name}: {len(df):,} events → {output_file.name} {comp_status} [{file_size_mb:.1f} MB]"
            
    except Exception as e:
        return f"✗ {root_file.name}: {e}"

def main():
    print("Starting parallel ROOT to pickle conversion...")
    
    # ========== CONFIGURATION ==========
    input_dir = Path("/eos/experiment/fcc/ee/analyses/case-studies/higgs/electron_yukawa/DataGenReduced/on-shell-electron/")
    output_dir = Path(".")
    tree_name = "events"
    max_events = 300_000
    
    # Compression setting
    # Set to False for faster loading during training (larger files)
    # Set to True for smaller files (slower loading)
    use_compression = False  # Changed to False for GPU training speed
    
    # ========== FILE SELECTION ==========
    # Specify which files to convert
    signal_files = [
        "wzp6_ee_Henueqq_ecm125.root",
        "wzp6_ee_Htaunutauqq_ecm125.root"
    ]
    
    ww_files = [
        "wzp6_ee_enueqq_ecm125.root",
        "wzp6_ee_munumuqq_ecm125.root",
        "wzp6_ee_taunutauqq_ecm125.root",
        "wzp6_ee_l1l2nunu_ecm125.root"
    ]
    
    zz_files = [
        "wzp6_ee_qq_ecm125.root",

        "wzp6_ee_tautau_ecm125.root",
        "wzp6_ee_tautaununu_ecm125.root",
        "wzp6_ee_tautauqq_ecm125.root",
        "p8_ee_ZZ_4tau_ecm125.root",

        "wzp6_ee_eenunu_ecm125.root",
        "wzp6_ee_eeqq_ecm125.root",

        "wzp6_ee_mumununu_ecm125.root",
        "wzp6_ee_mumuqq_ecm125.root",

        
        
        
    ]
    
    # Combine all files to convert
    files_to_convert = signal_files + ww_files + zz_files
    
    # Build full paths and check which exist
    root_files = []
    for fname in files_to_convert:
        full_path = input_dir / fname
        if full_path.exists():
            root_files.append(full_path)
        else:
            print(f"⚠️  File not found: {fname}")
    
    n_files = len(root_files)
    
    if n_files == 0:
        print("No ROOT files found!")
        return
    
    print(f"Found {n_files} ROOT file(s)")
    print(f"Compression: {'ENABLED (gzip)' if use_compression else 'DISABLED (faster loading)'}")
    
    # Determine optimal number of workers
    n_workers = max(1, mp.cpu_count() - 2)
    print(f"Using {n_workers} parallel workers\n")
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_file, rf, output_dir, tree_name, max_events, use_compression): rf
            for rf in root_files
        }
        
        # Process results as they complete
        total_size_mb = 0
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"[{i}/{n_files}] {result}")
            
            # Extract file size from result if successful
            if "MB]" in result:
                size_str = result.split("[")[1].split(" MB]")[0]
                total_size_mb += float(size_str)
    
    print(f"\n✓ All files processed!")
    print(f"Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")

if __name__ == "__main__":
    main()
