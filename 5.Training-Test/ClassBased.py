# Full 3‑Stage Pipeline Class
# -------------------------------------------
# Contains Stage 1 (ROOT→PKL), Stage 2 (Training), Stage 3 (Evaluation)
# and an example of how to run it at the bottom.

import os
import uproot
import awkward as ak
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")


class BDTYukawaPipeline:
    def __init__(self, config):
        self.config = config

        # Stage 1
        self.input_dir_stage1 = Path(config.get("input_dir_stage1", "."))
        self.output_dir_stage1 = Path(config.get("output_dir_stage1", "stage1_output"))
        self.tree_name_stage1 = config.get("tree_name_stage1", "events")
        self.max_events_stage1 = config.get("max_events_stage1", 100_000)

        # Stage 2
        self.train_signal = config.get("train_signal", [])
        self.train_background = config.get("train_background", [])
        self.model_output = config.get("model_output", "bdt_model.json")
        self.feature_importance_output = config.get("feature_importance_output", "feature_importance.csv")
        self.features = config.get("features", [])

        # Stage 3
        self.data_dir_stage3 = config.get("data_dir_stage3", "")
        self.signal_files_stage3 = config.get("signal_files_stage3", [])
        self.bg_files_stage3 = config.get("bg_files_stage3", [])
        self.eval_chunk = config.get("eval_chunk", 100_000)
        self.tree_name_stage3 = config.get("tree_name_stage3", "events")
        self.output_unbinned = config.get("output_unbinned", "bdt_scores_unbinned.pkl")


    # ---------------------------------------------------------
    # ---------------------- STAGE 1 --------------------------
    # ---------------------------------------------------------
    def stage1_convert_root_to_pickle(self):
        print("==== Stage 1: ROOT → Pickle ====")
        self.output_dir_stage1.mkdir(exist_ok=True)

        for root_file in self.input_dir_stage1.glob("*.root"):
            print(f"\nProcessing: {root_file.name}")
            try:
                with uproot.open(root_file) as f:
                    if self.tree_name_stage1 not in f:
                        print(f"  [!] No tree '{self.tree_name_stage1}', skipping.")
                        continue

                    tree = f[self.tree_name_stage1]
                    n_entries = tree.num_entries
                    print(f"  Found {n_entries:,} events.")

                    n_to_read = min(n_entries, self.max_events_stage1)
                    if n_entries > self.max_events_stage1:
                        print(f"  Reading first {self.max_events_stage1:,} events only.")

                    df = tree.arrays(library="pd", entry_stop=n_to_read).copy()
                    output_file = self.output_dir_stage1 / (root_file.stem + ".pkl")
                    df.to_pickle(output_file)

                    print(f"  Saved {len(df):,} events → {output_file}")

            except Exception as e:
                print(f"  [ERROR] Failed to process {root_file.name}: {e}")


    # ---------------------------------------------------------
    # ---------------------- STAGE 2 --------------------------
    # ---------------------------------------------------------
    def stage2_train_bdt(self):
        print("\n==== Stage 2: Training XGBoost BDT ====")

        # Load signal
        sig_df = [pd.read_pickle(f) for f in self.train_signal]
        sig_df = pd.concat(sig_df, ignore_index=True)
        sig_df["label"] = 1

        # Load background
        bg_df = [pd.read_pickle(f) for f in self.train_background]
        bg_df = pd.concat(bg_df, ignore_index=True)
        bg_df["label"] = 0

        df = pd.concat([sig_df, bg_df], ignore_index=True)

        # DMatrix
        X = df[self.features]
        y = df["label"]
        dtrain = xgb.DMatrix(X, label=y)

        params = {
            "max_depth": 6,
            "eta": 0.001,
            "objective": "binary:logistic",
            "eval_metric": "logloss"
        }

        bst = xgb.train(params, dtrain, num_boost_round=200)

        bst.save_model(self.model_output)
        print(f"📦 Saved model → {self.model_output}")

        # Feature importance
        importance = bst.get_score(importance_type="gain")
        imp_df = pd.DataFrame({"feature": list(importance.keys()), "gain": list(importance.values())})
        imp_df.to_csv(self.feature_importance_output, index=False)
        print(f"📄 Saved feature importance → {self.feature_importance_output}")


    # ---------------------------------------------------------
    # ---------------------- STAGE 3 --------------------------
    # ---------------------------------------------------------
    def load_root_in_chunks(self, file_list, step=100_000):
        tree_name = self.tree_name_stage3
        for arrays in uproot.iterate([f"{fn}:{tree_name}" for fn in file_list], self.features, step_size=step, library="ak"):
            yield ak.to_dataframe(arrays)

    def evaluate_file(self, file_path, model):
        preds_all = []
        for df in tqdm(self.load_root_in_chunks([file_path], step=self.eval_chunk), desc=f"{os.path.basename(file_path)}"):
            if df.empty:
                continue
            dmat = xgb.DMatrix(df[self.features])
            preds_all.append(model.predict(dmat))
        if preds_all:
            return np.concatenate(preds_all)
        return np.array([])

    def get_num_events(self, file_path):
        with uproot.open(file_path) as f:
            return f[self.tree_name_stage3].num_entries

    def base_process_name(self, filename):
        name = os.path.basename(filename).replace(".root", "")
        import re
        # Remove patterns: _1, _23, _chunk00, _chunk01, _chunk123
        name = re.sub(r"_(chunk\d+|\d+)$", "", name)
        return name + ".root"

    def stage3_evaluate(self):
        print("\n==== Stage 3: Evaluating Model ====")

        # Load model
        model = xgb.Booster()
        model.load_model(self.model_output)
        print(f"Loaded model: {self.model_output}")

        # Gather files
        file_list_sig = self.signal_files_stage3
        file_list_bg = self.bg_files_stage3

        def collect(files):
            results = {}
            for file_path in files:
                preds = self.evaluate_file(file_path, model)
                n_events = self.get_num_events(file_path)
                base_name = self.base_process_name(file_path)
                if base_name not in results:
                    results[base_name] = {"preds": [], "n_events": 0}
                results[base_name]["preds"].append(preds)
                results[base_name]["n_events"] += n_events
            return results

        print("Evaluating signal...")
        sig_res = collect(file_list_sig)

        print("Evaluating background...")
        bg_res = collect(file_list_bg)

        # Build output
        dfs = []
        for name, info in sig_res.items():
            preds = np.concatenate(info["preds"])
            df = pd.DataFrame({"process": name, "event_idx": np.arange(len(preds)), "bdt_score": preds})
            dfs.append(df)

        for name, info in bg_res.items():
            preds = np.concatenate(info["preds"])
            df = pd.DataFrame({"process": name, "event_idx": np.arange(len(preds)), "bdt_score": preds})
            dfs.append(df)

        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_pickle(self.output_unbinned)

        print(f"📦 Saved unbinned predictions → {self.output_unbinned}")
        print(f"Shape: {final_df.shape}")


# ---------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------
if __name__ == "__main__":
    config = {
        # ------- Stage 1 -------
        "input_dir_stage1": "./root_files/",
        "output_dir_stage1": "./pkl_stage1/",
        "tree_name_stage1": "events",
        "max_events_stage1": 100000,

        # ------- Stage 2 -------
        "train_signal": ["pkl_stage1/signal1.pkl"],
        "train_background": ["pkl_stage1/bg1.pkl", "pkl_stage1/bg2.pkl"],
        "model_output": "trained_model.json",
        "feature_importance_output": "feature_importance.csv",
        "features": ["Missing_Pt", "Jet1_Pt", "Jet2_Pt"],

        # ------- Stage 3 -------
        "signal_files_stage3": ["/eos/.../sig.root"],
        "bg_files_stage3": ["/eos/.../bg1.root", "/eos/.../bg2.root"],
        "output_unbinned": "bdt_scores_unbinned.pkl",
        "eval_chunk": 100000,
    }

    pipe = BDTYukawaPipeline(config)

    # Run any stage:
    # pipe.stage1_convert_root_to_pickle()
    # pipe.stage2_train_bdt()
    # pipe.stage3_evaluate()
