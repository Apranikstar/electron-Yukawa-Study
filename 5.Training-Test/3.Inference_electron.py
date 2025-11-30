########################################################################################################################################################################
#### if you have processes with various chunks at the end of process name add _1 _2 ...
import os
import uproot
import pandas as pd
import xgboost as xgb
import numpy as np
from tqdm import tqdm

# ------------------ CONFIG ------------------
data_dir = "/eos/experiment/fcc/ee/analyses/case-studies/higgs/electron_yukawa/DataGenReduced/on-shell-electron/"
tree_name = "events"

# GPU configuration
USE_GPU = True  # Set to False to use CPU
DEVICE = "cuda:0"  # GPU device to use (cuda:0, cuda:1, etc.)

signal_files = [
    os.path.join(data_dir, "wzp6_ee_Henueqq_ecm125.root"),
    os.path.join(data_dir, "wzp6_ee_Htaunutauqq_ecm125.root")
]

background_files = [
    os.path.join(data_dir, f) for f in os.listdir(data_dir)
    if f.endswith(".root")
    and f not in ["wzp6_ee_Henueqq_ecm125.root", "wzp6_ee_Htaunutauqq_ecm125.root"]
]

features = [
    "Iso_Photon_Phi",
    "Iso_Photon_Theta",
    "Iso_Photon_E",
    "Iso_Photon_CosTheta",
    "Iso_Photon_CosPhi",
    "Iso_Photons_No",

    "Iso_Electron_P",
    "Iso_Electron_Phi",
    "Iso_Electron_Theta",
    "Iso_Electron_E",
    "Iso_Electron_CosTheta",
    "Iso_Electron_CosPhi",
    "Iso_Electrons_No",

    "Missing_P",
    "Missing_E",
    "Missing_Pt",

    "EnuM",
    "Jets_InMa",

    "d23",
    "d34",

    "Jet_nconst0",
    "Jet_nconst1",
    
    "Jet1_P",
    "Jet1_Phi",
    "Jet1_M",
    "Jet1_E",
    "Jet1_Theta",
    "Jet1_CosTheta",
    "Jet1_CosPhi",

    "Jet2_P",
    "Jet2_Phi",
    "Jet2_M",
    "Jet2_E",
    "Jet2_Theta",
    "Jet2_CosTheta",
    "Jet2_CosPhi",

    "Max_JetsE",
    "Min_JetsE",
    "Jet1_charge",
    "Jet2_charge",

    "Jets_delR",
    "ILjet1_delR",
    "ILjet2_delR",
    "Max_DelRLJets",
    "Min_DelRLJets",

    "Jets_delphi",
    "ILjet1_delphi",
    "ILjet2_delphi",
    "Max_DelPhiLJets",
    "Min_DelPhiLJets",

    "Jets_deltheta",
    "Jets_angle",
    "ILjet1_angle",
    "ILjet2_angle",
    "Jets_cosangle",
    "ILjet1_cosangle",
    "ILjet2_cosangle",
    "Max_CosLJets",
    "Min_CosLJets",

    "Event_IM",
    "LJJ_M",
    "LJ1_M",
    "LJ2_M",
    "JJ_M",
    "JJ_E",

    "ljj_Phi",
    "jj_Phi",

    "Wl_M",
    "Wl_Theta",
    "Shell_M",
    "Off_M",
    "CosTheta_MaxjjW",
    "CosTheta_MinjjW",
    "expD",

    "Phi",
    "CosPhi",
    "Phi1",
    "CosPhi1",
    "PhiStar",
    "CosPhiStar",
    "ThetaStar",
    "CosThetaStar",
    "Theta1",
    "Costheta1",
    "Theta2",
    "Costheta2",
    "Planarity",
    "APlanarity",
    "Sphericity",
    "ASphericity",

    #"scoreMultiplyG",
    "scoreMultiplyU",
    "scoreMultiplyS",
    "scoreMultiplyC",
    "scoreMultiplyB",
    "scoreMultiplyT",
    "scoreMultiplyD",

    "scoreSumUD",
    "scoreSumDU",
    "scoreSumSC",
    "scoreSumCS",
    "scoreSumB",
    "scoreSumT",

    "scoreProdUD",
    "scoreProdDU",
    "scoreProdSC",
    "scoreProdCS",
]

# Process 1M events per chunk
chunk_size = 1_000_000

# ------------------ FUNCTIONS ------------------
def flatten_awkward_columns(df):
    """
    Convert any awkward/non-numeric columns to numeric values.
    Matches the training preprocessing: takes mean of arrays, 0 otherwise.
    """
    for col in df.columns:
        # Check if column is non-numeric (including awkward dtype)
        dtype_str = str(df[col].dtype)
        if not pd.api.types.is_numeric_dtype(df[col]) or dtype_str == 'awkward':
            # Apply the same logic as training: mean of arrays, or scalar/0
            df[col] = df[col].apply(
                lambda x: np.mean(x) if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0
                else (x if isinstance(x, (int, float, np.number)) else 0)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def load_root_in_chunks(file_list, step=1_000_000):
    """Iterate over ROOT files in chunks and return DataFrames."""
    for df in uproot.iterate(
        [f"{fn}:{tree_name}" for fn in file_list],
        features,
        step_size=step,
        library="pd"
    ):
        # Flatten any awkward array columns to match training data format
        df = flatten_awkward_columns(df)
        yield df

def evaluate_file(file_path, model, features, step=1_000_000, use_gpu=False):
    """Evaluate BDT model on a ROOT file and return predictions."""
    preds_all = []

    for df in tqdm(load_root_in_chunks([file_path], step=step),
                   desc=f"Processing {os.path.basename(file_path)}"):
        if df.empty:
            continue
        
        # Verify all columns are numeric before creating DMatrix
        non_numeric = [c for c in df.columns if str(df[c].dtype) == 'awkward' or not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            print(f"ERROR: Non-numeric columns still present: {non_numeric}")
            print("Column dtypes:")
            for c in non_numeric:
                print(f"  {c}: {df[c].dtype}")
            raise ValueError("Cannot create DMatrix with non-numeric columns")
        
        # Create DMatrix with GPU support
        if use_gpu:
            dmat = xgb.DMatrix(df[features], enable_categorical=False)
            dmat.set_info(feature_names=features)
        else:
            dmat = xgb.DMatrix(df[features])
        
        preds = model.predict(dmat)
        preds_all.append(preds)

    if not preds_all:
        return np.array([]).reshape(0, 5)  # Return empty array with correct shape for 5 classes

    return np.vstack(preds_all)

def get_num_events(file_path):
    """Return total number of entries in the ROOT tree."""
    with uproot.open(file_path) as f:
        return f[tree_name].num_entries

def base_process_name(filename):
    """Strip trailing _1, _2, etc. to merge subsamples."""
    name = os.path.basename(filename)
    name = name.replace(".root", "")
    if name.split("_")[-1].isdigit():
        name = "_".join(name.split("_")[:-1])
    return name + ".root"

# ------------------ LOAD MODEL ------------------
print("=" * 60)
print("XGBoost 5-Class GPU Inference Pipeline")
print("=" * 60)

# Load model
bst = xgb.Booster()
bst.load_model("multiclass_5class_model.json")

# Configure model for GPU inference
if USE_GPU:
    bst.set_param({"device": DEVICE})
    print(f"✅ Loaded model with GPU support ({DEVICE})")
    print(f"   Using batch size: {chunk_size:,} events per chunk")
else:
    print("✅ Loaded model (CPU mode)")
    print(f"   Using batch size: {chunk_size:,} events per chunk")

# Verify GPU availability if requested
if USE_GPU:
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("   WARNING: GPU requested but CUDA not available, falling back to CPU")
            USE_GPU = False
    except ImportError:
        print("   WARNING: PyTorch not found, cannot verify GPU. Proceeding anyway...")

print()

# ------------------ EVALUATE ------------------
def collect_results(file_list, model, use_gpu=False):
    results = {}
    for file_path in file_list:
        preds = evaluate_file(file_path, model, features, chunk_size, use_gpu=use_gpu)
        n_events = get_num_events(file_path)
        base_name = base_process_name(file_path)

        if base_name not in results:
            results[base_name] = {"preds": [], "n_events": 0}
        results[base_name]["preds"].append(preds)
        results[base_name]["n_events"] += n_events
    return results

print("=== Evaluating Signal Samples ===")
sig_results = collect_results(signal_files, bst, use_gpu=USE_GPU)

print("\n=== Evaluating Background Samples ===")
bg_results = collect_results(background_files, bst, use_gpu=USE_GPU)

# ------------------ SAVE UNBINNED DISTRIBUTIONS ------------------
output_dir = "bdt_distributions"
os.makedirs(output_dir, exist_ok=True)

def make_unbinned_df(preds, process_name, n_events, cross_section=0.0):
    """Create a DataFrame of raw BDT predictions for one process with metadata."""
    df = pd.DataFrame({
        "process": process_name,
        "cross_section": cross_section,
        "total_events_in_file": n_events,
        "event_idx": np.arange(len(preds)),
        "bdt_signal": preds[:, 0],          # Class 0: Signal probability
        "bdt_ww": preds[:, 1],              # Class 1: WW probability
        "bdt_zz_leptonic": preds[:, 2],     # Class 2: ZZ_leptonic probability
        "bdt_zz_2lep2j": preds[:, 3],       # Class 3: ZZ_2lep2j probability
        "bdt_zstar": preds[:, 4]            # Class 4: Zstar probability
    })
    return df

dfs = []

print("\n=== Building final unbinned DataFrame with metadata ===")
for name, data in sig_results.items():
    preds = np.vstack(data["preds"])
    df = make_unbinned_df(preds, name, data["n_events"], cross_section=0.0)
    dfs.append(df)

for name, data in bg_results.items():
    preds = np.vstack(data["preds"])
    df = make_unbinned_df(preds, name, data["n_events"], cross_section=0.0)
    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)

# ------------------ SAVE TO PICKLE ------------------
output_pkl = os.path.join(output_dir, "bdt_scores_unbinned_5class.pkl")
final_df.to_pickle(output_pkl)

print(f"\n📦 Final DataFrame saved → {output_pkl}")
print(f"✅ Shape: {final_df.shape}")
print("\nProcesses in final DataFrame:")
print(final_df['process'].unique())
print("\nPreview:")
print(final_df.head())
print("\nVerify probabilities sum to 1.0:")
prob_sum = final_df[['bdt_signal', 'bdt_ww', 'bdt_zz_leptonic', 'bdt_zz_2lep2j', 'bdt_zstar']].sum(axis=1)
print(f"  Min sum: {prob_sum.min():.6f}")
print(f"  Max sum: {prob_sum.max():.6f}")
print(f"  Mean sum: {prob_sum.mean():.6f}")
print("\n" + "=" * 60)
print("Inference complete!")
print("=" * 60)
