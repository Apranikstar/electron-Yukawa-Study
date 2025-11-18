import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==== CONFIGURATION SECTION ====
# All inputs and outputs are defined here
CONFIG = {
    # Data paths
    "data_dir": ".",
    "signal_files": [
        "wzp6_ee_Henueqq_ecm125.pkl",
        "wzp6_ee_Htaunutauqq_ecm125.pkl",
    ],
    
    # Output paths
    "model_output": "on-shell-electron.json",
    "feature_importance_csv": "feature_importance.csv",
    "feature_importance_plot": "feature_importance.png",
    "training_log": "training_log.txt",
    
    # Training parameters
    "test_size": 0.2,
    "random_state": 42,
    "xgb_params": {
        "objective": "binary:logistic",
        "max_depth": 6,
        "eta": 0.001,
        "eval_metric": "auc",
        "tree_method": "hist",
        "nthread": 64
    },
    "num_boost_round": 30000,
    "early_stopping_rounds": 100,
    "verbose_eval": 50,
    
    # Feature importance settings
    "importance_type": "gain",  # Options: 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
    "top_n_features_plot": 30,  # Number of top features to show in plot
}

# Feature list
features = [
    "IsoMuonNum",
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
    "IsoElectron_3p",
    "Iso_Electron_P",
    "Iso_Electron_Pt",
    "Iso_Electron_Eta",
    "Iso_Electron_Phi",
    "Iso_Electron_Rapidity",
    "Iso_Electron_Theta",
    "Iso_Electron_M",
    "Iso_Electron_Mt",
    "Iso_Electron_E",
    "Iso_Electron_Et",
    "Iso_Electron_CosTheta",
    "Iso_Electron_CosPhi",
    "Iso_Electrons_No",
    "Iso_Electron_Charge",
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
    "EnuM",
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
    "mela",
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

# ==== HELPER FUNCTIONS ====
def load_pickle_to_df(file_list, label_value, data_dir):
    dfs = []
    for fn in file_list:
        full_path = os.path.join(data_dir, fn) if not os.path.isabs(fn) else fn
        print(f"  Checking {full_path} ...")
        if not os.path.exists(full_path):
            print(f"  [ERROR] Missing file: {full_path}")
            continue
        try:
            print(f"  Reading {full_path}")
            df = pd.read_pickle(full_path)
            print(f"    Shape: {df.shape}")
            df["label"] = label_value
            dfs.append(df)
        except Exception as e:
            print(f"  [ERROR] Failed to read {full_path}: {e}")
    if len(dfs) == 0:
        raise RuntimeError("No valid DataFrames loaded!")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  --> Combined DataFrame shape: {combined.shape}")
    return combined

def save_feature_importance(bst, feature_names, config):
    """Save feature importance to CSV and create visualization"""
    importance_type = config["importance_type"]
    
    # Get feature importance
    importance_dict = bst.get_score(importance_type=importance_type)
    
    # Create DataFrame with all features (including those with 0 importance)
    importance_data = []
    for fname in feature_names:
        importance_data.append({
            "feature": fname,
            "importance": importance_dict.get(fname, 0)
        })
    
    df_importance = pd.DataFrame(importance_data)
    df_importance = df_importance.sort_values("importance", ascending=False)
    df_importance["rank"] = range(1, len(df_importance) + 1)
    
    # Save to CSV
    csv_path = config["feature_importance_csv"]
    df_importance.to_csv(csv_path, index=False)
    print(f"\n=== Feature importance saved to {csv_path} ===")
    print(f"\nTop 10 most important features ({importance_type}):")
    print(df_importance.head(10).to_string(index=False))
    
    # Create visualization
    top_n = min(config["top_n_features_plot"], len(df_importance))
    top_features = df_importance.head(top_n)
    
    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    plt.barh(range(top_n), top_features["importance"].values)
    plt.yticks(range(top_n), top_features["feature"].values)
    plt.xlabel(f"Importance ({importance_type})")
    plt.title(f"Top {top_n} Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path = config["feature_importance_plot"]
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Feature importance plot saved to {plot_path}")
    plt.close()
    
    return df_importance

# ==== MAIN ====
def main():
    print("="*60)
    print("XGBoost Training Pipeline")
    print("="*60)
    print("\n=== Configuration ===")
    print(f"Data directory: {CONFIG['data_dir']}")
    print(f"Model output: {CONFIG['model_output']}")
    print(f"Feature importance CSV: {CONFIG['feature_importance_csv']}")
    print(f"Feature importance plot: {CONFIG['feature_importance_plot']}")
    print(f"Number of features: {len(features)}")
    
    # Collect all pickle files
    print("\n=== Collecting all pickle files ===")
    all_files = sorted(glob.glob(os.path.join(CONFIG['data_dir'], "*.pkl")))
    print(f"Found {len(all_files)} .pkl files")
    
    # Separate signal and background
    signal_full_paths = [os.path.join(CONFIG['data_dir'], f) for f in CONFIG['signal_files']]
    background_files = [f for f in all_files if f not in signal_full_paths]
    print(f"Signal files: {len(CONFIG['signal_files'])}, Background files: {len(background_files)}")
    
    # Load data
    print("\n=== Loading SIGNAL ===")
    df_signal = load_pickle_to_df(CONFIG['signal_files'], label_value=1, data_dir=CONFIG['data_dir'])
    
    print("\n=== Loading BACKGROUND ===")
    df_background = load_pickle_to_df(background_files, label_value=0, data_dir=CONFIG['data_dir'])
    
    # Merge
    print("\n=== Merging dataframes ===")
    df_all = pd.concat([df_signal, df_background], ignore_index=True)
    print(f"Total events loaded: {len(df_all):,}")
    print(f"Memory usage: {df_all.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    
    # Clean columns
    print("\n=== Checking column data types ===")
    bad_cols = [c for c in df_all.columns if not pd.api.types.is_numeric_dtype(df_all[c])]
    print(f"Non-numeric columns detected: {bad_cols}")
    
    if bad_cols:
        for c in bad_cols:
            print(f"  Cleaning column: {c}")
            df_all[c] = df_all[c].apply(
                lambda x: np.mean(x) if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0
                else (x if isinstance(x, (int, float, np.number)) else 0)
            )
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce").fillna(0)
        print("  -> All columns converted to numeric")
    
    # Check feature coverage
    missing_features = [f for f in features if f not in df_all.columns]
    if missing_features:
        print(f"[WARNING] Missing features: {missing_features}")
    else:
        print("All expected features found!")
    
    # Split data
    print("\n=== Splitting data ===")
    X = df_all[features].copy()
    y = df_all["label"]
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state'], 
        stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # XGBoost Training
    print("\n=== Starting XGBoost training ===")
    print(f"Parameters: {CONFIG['xgb_params']}")
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)
    
    evals = [(dtrain, "train"), (dtest, "test")]
    
    bst = xgb.train(
        CONFIG['xgb_params'],
        dtrain,
        num_boost_round=CONFIG['num_boost_round'],
        evals=evals,
        early_stopping_rounds=CONFIG['early_stopping_rounds'],
        verbose_eval=CONFIG['verbose_eval']
    )
    
    # Save model
    print("\n=== Training complete! ===")
    bst.save_model(CONFIG['model_output'])
    print(f"Model saved as {CONFIG['model_output']}")
    
    # Save feature importance
    print("\n=== Analyzing feature importance ===")
    df_importance = save_feature_importance(bst, features, CONFIG)
    
    print("\n=== Pipeline complete! ===")
    print(f"Outputs saved:")
    print(f"  - Model: {CONFIG['model_output']}")
    print(f"  - Feature importance CSV: {CONFIG['feature_importance_csv']}")
    print(f"  - Feature importance plot: {CONFIG['feature_importance_plot']}")

if __name__ == "__main__":
    main()
