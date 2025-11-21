########################################################################################################################################################################
#### if you have processes with various chunks at the end of process name add _1 _2 ...
import os
import uproot
import pandas as pd
import xgboost as xgb
import numpy as np
from tqdm import tqdm

# ------------------ CONFIG ------------------
data_dir = "/eos/experiment/fcc/ee/analyses/case-studies/higgs/electron_yukawa/DataGen/off-shell-muon"
tree_name = "events"

signal_files = [
    os.path.join(data_dir, "wzp6_ee_Hqqmunumu_ecm125.root"),
    os.path.join(data_dir, "wzp6_ee_Hqqtaunutau_ecm125.root")
]

background_files = [
    os.path.join(data_dir, f) for f in os.listdir(data_dir)
    if f.endswith(".root")
    and f not in ["wzp6_ee_Hqqmunumu_ecm125.root", "wzp6_ee_Hqqtaunutau_ecm125.root"]
]
features = [

            "Iso_Electrons_No",
            "Missing_Pt",
            "Iso_Photon_P",
            "Iso_Photon_Pt",
            "Iso_Photon_Eta",
            "Iso_Photon_Phi",
            "Iso_Photon_Rapidity",
            "Iso_Photon_Theta",
            "Iso_Photon_M",
            "Iso_Photon_Mt",
            "Iso_Photon_E",
            "Iso_Photon_Et",
            "Iso_Photon_CosTheta",
            "Iso_Photon_CosPhi",
            "Iso_Photons_No",
            "IsoMuon_3p",
            "Iso_Muon_P",
            "Iso_Muon_Pt",
            "Iso_Muon_Eta",
            "Iso_Muon_Phi",
            "Iso_Muon_Rapidity",
            "Iso_Muon_Theta",
            "Iso_Muon_M",
            "Iso_Muon_Mt",
            "Iso_Muon_E",
            "Iso_Muon_Et",
            "Iso_Muon_CosTheta",
            "Iso_Muon_CosPhi",
            "IsoMuonNum",
            "Iso_Muon_Charge",
            "Missing_P",
            "Missing_Eta",
            "Missing_Phi",
            "Missing_Rapidity",
            "Missing_Theta",
            "Missing_M",
            "Missing_Mt",
            "Missing_E",
            "Missing_Et",
            "Missing_CosTheta",
            "Missing_CosPhi",
            "MunuM",
            "Jets_InMa",
            "d23",
            "d34",
            "Jets_charge",
            "Jet_nconst0",
            "Jet_nconst1",
            "displacementdz0",
            "displacementdxy0",
            "displacementdz1",
            "displacementdxy1",
            "Jet1_P3",
            "Jet1_P",
            "Jet1_Pt",
            "Jet1_Eta",
            "Jet1_Rapidity",
            "Jet1_Phi",
            "Jet1_M",
            "Jet1_Mt",
            "Jet1_E",
            "Jet1_Et",
            "Jet1_Theta",
            "Jet1_CosTheta",
            "Jet1_CosPhi",
            "Jet2_P3",
            "Jet2_P",
            "Jet2_Pt",
            "Jet2_Eta",
            "Jet2_Rapidity",
            "Jet2_Phi",
            "Jet2_M",
            "Jet2_Mt",
            "Jet2_E",
            "Jet2_Et",
            "Jet2_Theta",
            "Jet2_CosTheta",
            "Jet2_CosPhi",
            "Max_JetsPT",
            "Min_JetsPT",
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
            "Jets_deleta",
            "ILjet1_deleta",
            "ILjet2_deleta",
            "Max_DelEtaLJets",
            "Min_DelEtaLJets",
            "Jets_delrapi",
            "ILjet1_delrapi",
            "ILjet2_delrapi",
            "Max_DelyLJets",
            "Min_DelyLJets",
            "Jets_deltheta",
            "Jets_angle",
            "ILjet1_angle",
            "ILjet2_angle",
            "Jets_cosangle",
            "ILjet1_cosangle",
            "ILjet2_cosangle",
            "Max_CosLJets",
            "Min_CosLJets",
            "HT",
            "Event_IM",
            "LJJ_M",
            "LJJ_Mt",
            "LJ1_M",
            "LJ1_Mt",
            "LJ2_M",
            "LJ2_Mt",
            "Lnu_M",
            "JJ_M",
            "JJ_Mt",
            "JJ_E",
            "lj1_PT",
            "lj2_PT",
            "jj_PT",
            "ljj_y",
            "jj_y",
            "lj1_y",
            "lj2_y",
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


            "scoreG1",
            "scoreG2",
            "scoreU1",
            "scoreU2",
            "scoreS1",
            "scoreS2",
            "scoreC1",
            "scoreC2",
            "scoreB1",
            "scoreB2",
            "scoreT1",
            "scoreT2",
            "scoreD1",
            "scoreD2",
            "scoreSumG",
            "scoreSumU",
            "scoreSumS",
            "scoreSumC",
            "scoreSumB",
            "scoreSumT",
            "scoreSumD",
            "scoreMultiplyG",
            "scoreMultiplyU",
            "scoreMultiplyS",
            "scoreMultiplyC",
            "scoreMultiplyB",
            "scoreMultiplyT",
            "scoreMultiplyD",

]

chunk_size = 100_000

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
            #print(f"  Flattening column: {col} (dtype: {dtype_str})")
            # Apply the same logic as training: mean of arrays, or scalar/0
            df[col] = df[col].apply(
                lambda x: np.mean(x) if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0
                else (x if isinstance(x, (int, float, np.number)) else 0)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def load_root_in_chunks(file_list, step=100_000):
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

def evaluate_file(file_path, model, features, step=100_000):
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
        
        dmat = xgb.DMatrix(df[features])
        preds = model.predict(dmat)
        preds_all.append(preds)

    if not preds_all:
        return np.array([])

    return np.concatenate(preds_all)

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
bst = xgb.Booster()
bst.load_model("off-shell-muon.json")
print("✅ Loaded model from off-shell-muon.json")

# ------------------ EVALUATE ------------------
def collect_results(file_list, model):
    results = {}
    for file_path in file_list:
        preds = evaluate_file(file_path, model, features, chunk_size)
        n_events = get_num_events(file_path)
        base_name = base_process_name(file_path)

        if base_name not in results:
            results[base_name] = {"preds": [], "n_events": 0}
        results[base_name]["preds"].append(preds)
        results[base_name]["n_events"] += n_events
    return results

print("\n=== Evaluating Signal Samples ===")
sig_results = collect_results(signal_files, bst)

print("\n=== Evaluating Background Samples ===")
bg_results = collect_results(background_files, bst)

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
        "bdt_score": preds
    })
    return df

dfs = []

print("\n=== Building final unbinned DataFrame with metadata ===")
for name, data in sig_results.items():
    preds = np.concatenate(data["preds"])
    df = make_unbinned_df(preds, name, data["n_events"], cross_section=0.0)
    dfs.append(df)

for name, data in bg_results.items():
    preds = np.concatenate(data["preds"])
    df = make_unbinned_df(preds, name, data["n_events"], cross_section=0.0)
    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)

# ------------------ SAVE TO PICKLE ------------------
output_pkl = os.path.join(output_dir, "bdt_scores_unbinned.pkl")
final_df.to_pickle(output_pkl)

print(f"\n📦 Final DataFrame saved → {output_pkl}")
print(f"✅ Shape: {final_df.shape}")
print("\nProcesses in final DataFrame:")
print(final_df['process'].unique())
print("\nPreview:")
print(final_df.head())
