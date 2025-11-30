import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==== CONFIGURATION SECTION ====
CONFIG = {
    # Data paths
    "data_dir": ".",
    "signal_files": [
        "wzp6_ee_Henueqq_ecm125.pkl",
        "wzp6_ee_Htaunutauqq_ecm125.pkl",
    ],
    "ww_files": [
        "wzp6_ee_enueqq_ecm125.pkl",
        "wzp6_ee_munumuqq_ecm125.pkl",
        "wzp6_ee_taunutauqq_ecm125.pkl",
        "wzp6_ee_l1l2nunu_ecm125.pkl"
    ],
    "zz_files_leptonic": [
        "wzp6_ee_tautaununu_ecm125.pkl",
        "wzp6_ee_eenunu_ecm125.pkl",
        "wzp6_ee_mumununu_ecm125.pkl",
        "p8_ee_ZZ_4tau_ecm125.pkl",
    ],
    "zz_files_2lep2j": [
        "wzp6_ee_tautauqq_ecm125.pkl",
        "wzp6_ee_mumuqq_ecm125.pkl",
        "wzp6_ee_eeqq_ecm125.pkl"
    ],
    "zstar_files": [
        "wzp6_ee_tautau_ecm125.pkl",
        "wzp6_ee_qq_ecm125.pkl",
    ],
    
    # Output paths
    "model_output": "multiclass_5class_model.json",
    "feature_importance_csv": "feature_importance_5class.csv",
    "feature_importance_plot": "feature_importance_5class.png",
    "training_log": "training_log_5class.txt",
    "confusion_matrix_plot": "confusion_matrix_5class.png",
    "roc_curve_plot": "roc_curves_5class.png",
    
    # Training parameters
    "test_size": 0.2,
    "random_state": 42,
    "xgb_params": {
        "objective": "multi:softprob",  # Multi-class classification
        "num_class": 5,  # Signal (0), WW (1), ZZ_leptonic (2), ZZ_2lep2j (3), Zstar (4)
        "max_depth": 6,
        "eta": 0.001,
        "eval_metric": "mlogloss",  # Multi-class log loss
        "tree_method": "hist",  # Use "hist" for CPU or "gpu_hist" for GPU
        "device": "cuda",  # Use GPU (change to "cpu" for CPU training)
        "nthread": 64
    },
    "num_boost_round": 30000,
    "early_stopping_rounds": 100,
    "verbose_eval": 50,
    
    # Feature importance settings
    "importance_type": "gain",
    "top_n_features_plot": 30,
}

# Feature list - only the specified features
features = [            "Iso_Photon_Phi",
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

# Class labels mapping
CLASS_LABELS = {
    0: "Signal",
    1: "WW",
    2: "ZZ_leptonic",
    3: "ZZ_2lep2j",
    4: "Zstar"
}

# ==== HELPER FUNCTIONS ====
def load_pickle_to_df(file_list, label_value, data_dir, class_name):
    """Load pickle files and assign label"""
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
        print(f"  [WARNING] No valid DataFrames loaded for {class_name}!")
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  --> Combined {class_name} DataFrame shape: {combined.shape}")
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
    plt.title(f"Top {top_n} Feature Importance (5-Class)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path = config["feature_importance_plot"]
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Feature importance plot saved to {plot_path}")
    plt.close()
    
    return df_importance

def plot_confusion_matrix(y_true, y_pred, config):
    """Create and save confusion matrix plot"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix (each row sums to 1)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title('Confusion Matrix (5-Class) - Normalized')
    plt.colorbar()
    
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, [CLASS_LABELS[i] for i in range(5)], rotation=45, ha='right')
    plt.yticks(tick_marks, [CLASS_LABELS[i] for i in range(5)])
    
    # Add text annotations with both normalized values and counts
    thresh = 0.5
    for i in range(5):
        for j in range(5):
            plt.text(j, i, f'{cm_normalized[i, j]:.3f}\n({cm[i, j]})',
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black",
                    fontsize=8)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    plot_path = config["confusion_matrix_plot"]
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix saved to {plot_path}")
    plt.close()

def plot_roc_curves(y_true, y_pred_proba, config):
    """Create and save ROC curves for Signal vs each background class"""
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=(9, 7))
    
    # Signal is class 0, we want to plot Signal vs each background
    # Create binary labels: Signal (1) vs each background class (0)
    signal_mask = (y_true == 0)
    signal_proba = y_pred_proba[:, 0]  # Probability of being signal
    
    # Define colors for each background
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    background_classes = [1, 2, 3, 4]  # WW, ZZ_leptonic, ZZ_2lep2j, Zstar
    
    for idx, bg_class in enumerate(background_classes):
        # Create binary problem: Signal vs this specific background
        mask = (y_true == 0) | (y_true == bg_class)
        y_binary = (y_true[mask] == 0).astype(int)  # 1 for signal, 0 for background
        y_score = signal_proba[mask]
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, color=colors[idx], lw=2,
                label=f'Signal vs {CLASS_LABELS[bg_class]} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Signal vs Backgrounds', fontsize=14)
    plt.legend(loc="lower right", frameon=False, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = config["roc_curve_plot"]
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"ROC curves saved to {plot_path}")
    plt.close()

# ==== MAIN ====
def main():
    print("="*60)
    print("XGBoost 5-Class Training Pipeline")
    print("Signal / WW / ZZ_leptonic / ZZ_2lep2j / Zstar Classification")
    print("="*60)
    print("\n=== Configuration ===")
    print(f"Data directory: {CONFIG['data_dir']}")
    print(f"Model output: {CONFIG['model_output']}")
    print(f"Number of features: {len(features)}")
    print(f"Classes: {CLASS_LABELS}")
    
    # Load data for each class
    print("\n=== Loading SIGNAL (Class 0) ===")
    df_signal = load_pickle_to_df(CONFIG['signal_files'], label_value=0, 
                                   data_dir=CONFIG['data_dir'], class_name="Signal")
    
    print("\n=== Loading WW Background (Class 1) ===")
    df_ww = load_pickle_to_df(CONFIG['ww_files'], label_value=1, 
                              data_dir=CONFIG['data_dir'], class_name="WW")
    
    print("\n=== Loading ZZ Leptonic Background (Class 2) ===")
    df_zz_lep = load_pickle_to_df(CONFIG['zz_files_leptonic'], label_value=2, 
                                   data_dir=CONFIG['data_dir'], class_name="ZZ_leptonic")
    
    print("\n=== Loading ZZ 2lep2j Background (Class 3) ===")
    df_zz_2lep2j = load_pickle_to_df(CONFIG['zz_files_2lep2j'], label_value=3, 
                                      data_dir=CONFIG['data_dir'], class_name="ZZ_2lep2j")
    
    print("\n=== Loading Zstar Background (Class 4) ===")
    df_zstar = load_pickle_to_df(CONFIG['zstar_files'], label_value=4, 
                                  data_dir=CONFIG['data_dir'], class_name="Zstar")
    
    # Merge all classes
    print("\n=== Merging dataframes ===")
    dfs_to_merge = [df for df in [df_signal, df_ww, df_zz_lep, df_zz_2lep2j, df_zstar] if df is not None]
    
    if len(dfs_to_merge) == 0:
        raise RuntimeError("No valid dataframes loaded!")
    
    df_all = pd.concat(dfs_to_merge, ignore_index=True)
    print(f"Total events loaded: {len(df_all):,}")
    print(f"Class distribution:")
    for label, name in CLASS_LABELS.items():
        count = (df_all['label'] == label).sum()
        print(f"  {name} (Class {label}): {count:,} events ({count/len(df_all)*100:.1f}%)")
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
        print("Available columns:", list(df_all.columns))
        raise ValueError(f"Cannot proceed with missing features: {missing_features}")
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
    print("Train class distribution:")
    for label, name in CLASS_LABELS.items():
        count = (y_train == label).sum()
        print(f"  {name}: {count:,} events")
    
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
    
    # Make predictions for evaluation
    print("\n=== Evaluating model ===")
    y_pred_proba = bst.predict(dtest)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate accuracies
    from sklearn.metrics import accuracy_score, classification_report
    overall_acc = accuracy_score(y_test, y_pred)
    print(f"Overall accuracy: {overall_acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=[CLASS_LABELS[i] for i in range(5)]))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, CONFIG)
    
    # Plot ROC curves
    plot_roc_curves(y_test, y_pred_proba, CONFIG)
    
    # Save feature importance
    print("\n=== Analyzing feature importance ===")
    df_importance = save_feature_importance(bst, features, CONFIG)
    
    print("\n=== Pipeline complete! ===")
    print(f"Outputs saved:")
    print(f"  - Model: {CONFIG['model_output']}")
    print(f"  - Feature importance CSV: {CONFIG['feature_importance_csv']}")
    print(f"  - Feature importance plot: {CONFIG['feature_importance_plot']}")
    print(f"  - Confusion matrix: {CONFIG['confusion_matrix_plot']}")
    print(f"  - ROC curves: {CONFIG['roc_curve_plot']}")

if __name__ == "__main__":
    main()
