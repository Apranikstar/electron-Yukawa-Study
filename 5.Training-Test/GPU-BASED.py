#!/usr/bin/env python3
"""
BDTYukawaPipeline GPU version (single-file)
- Stage 1: ROOT -> cuDF -> Parquet
- Stage 2: Train XGBoost on GPU
- Stage 3: Evaluate ROOT files in chunks on GPU, save unbinned scores (Parquet or Pickle)

Requirements:
 - uproot
 - awkward
 - cudf (RAPIDS)
 - cupy
 - xgboost (GPU-enabled)
 - numpy, pandas (only minimal, for CPU fallbacks)
 - tqdm

Author: adapted for user request
"""

import os
import re
from pathlib import Path
from tqdm import tqdm

import uproot
import awkward as ak
import numpy as np
import pandas as pd

# RAPIDS / GPU libraries
try:
    import cudf
    import cupy as cp
except Exception as e:
    raise ImportError("This script requires cudf and cupy (RAPIDS). Install them before running.") from e

import xgboost as xgb


class BDTYukawaPipelineGPU:
    def __init__(self, config):
        self.config = config

        # Stage 1
        self.input_dir_stage1 = Path(config.get("input_dir_stage1", "."))
        self.output_dir_stage1 = Path(config.get("output_dir_stage1", "stage1_output_parquet"))
        self.tree_name_stage1 = config.get("tree_name_stage1", "events")
        self.max_events_stage1 = config.get("max_events_stage1", 100_000)
        # prefer parquet for GPU-friendly IO
        self.stage1_file_ext = config.get("stage1_file_ext", ".parquet")

        # Stage 2
        self.train_signal = config.get("train_signal", [])
        self.train_background = config.get("train_background", [])
        self.model_output = config.get("model_output", "bdt_model_gpu.json")
        self.feature_importance_output = config.get("feature_importance_output", "feature_importance_gpu.csv")
        self.features = config.get("features", [])

        # Stage 3
        self.data_dir_stage3 = config.get("data_dir_stage3", "")
        self.signal_files_stage3 = config.get("signal_files_stage3", [])
        self.bg_files_stage3 = config.get("bg_files_stage3", [])
        self.eval_chunk = config.get("eval_chunk", 100_000)
        self.tree_name_stage3 = config.get("tree_name_stage3", "events")
        self.output_unbinned = config.get("output_unbinned", "bdt_scores_unbinned.parquet")

        # sanity checks
        if not self.features:
            raise ValueError("No features specified in config['features']")

    # ---------------------- STAGE 1 --------------------------
    def stage1_convert_root_to_parquet(self):
        print("==== Stage 1: ROOT → cuDF Parquet ====")
        self.output_dir_stage1.mkdir(parents=True, exist_ok=True)

        root_files = sorted(self.input_dir_stage1.glob("*.root"))
        if not root_files:
            print(f"[!] No .root files found in {self.input_dir_stage1}")
            return

        for root_file in root_files:
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

                    # read arrays as numpy (avoid awkward -> pandas intermediate)
                    arrays = tree.arrays(self.features, entry_stop=n_to_read, library="np")
                    # arrays is a dict of numpy arrays per field
                    # convert to cuDF DataFrame (on GPU)
                    # cudf.DataFrame accepts dict of array-like (numpy or cupy)
                    gdf = cudf.DataFrame({k: cp.asarray(v) for k, v in arrays.items()})

                    out_file = self.output_dir_stage1 / (root_file.stem + self.stage1_file_ext)
                    # write parquet (fast and GPU-friendly)
                    gdf.to_parquet(out_file)
                    print(f"  Saved {len(gdf):,} events → {out_file}")

            except Exception as e:
                print(f"  [ERROR] Failed to process {root_file.name}: {e}")

    # ---------------------- STAGE 2 --------------------------
    def stage2_train_bdt(self, num_rounds=200, params_override=None):
        print("\n==== Stage 2: Training XGBoost (GPU) ====")

        if not self.train_signal or not self.train_background:
            raise ValueError("train_signal and train_background must be specified and non-empty lists")

        # Load signal (cuDF parquet)
        sig_gdfs = []
        for f in self.train_signal:
            print(f"  Loading signal: {f}")
            sig_gdfs.append(cudf.read_parquet(f))
        sig_df = cudf.concat(sig_gdfs, ignore_index=True)
        sig_df["label"] = 1

        # Load background
        bg_gdfs = []
        for f in self.train_background:
            print(f"  Loading background: {f}")
            bg_gdfs.append(cudf.read_parquet(f))
        bg_df = cudf.concat(bg_gdfs, ignore_index=True)
        bg_df["label"] = 0

        full = cudf.concat([sig_df, bg_df], ignore_index=True)
        # select features and labels
        X = full[self.features]
        y = full["label"]

        # Build DMatrix - modern xgboost supports cuDF input directly with GPU build
        try:
            dtrain = xgb.DMatrix(X, label=y)
        except Exception:
            # fallback: convert to CuPy arrays explicitly
            print("  Warning: xgboost.DMatrix did not accept cuDF directly, converting to cupy arrays.")
            X_cp = [cp.asarray(X[col].to_numpy()) for col in X.columns]
            X_cp = cp.stack(X_cp, axis=1)
            dtrain = xgb.DMatrix(X_cp, label=cp.asarray(y.to_numpy()))

        params = {
            "max_depth": 6,
            "eta": 0.001,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
        }
        if params_override:
            params.update(params_override)

        print("  Training on GPU...")
        bst = xgb.train(params, dtrain, num_boost_round=num_rounds)

        bst.save_model(self.model_output)
        print(f"📦 Saved model → {self.model_output}")

        # Feature importance (gain)
        try:
            importance = bst.get_score(importance_type="gain")
            imp_df = pd.DataFrame({"feature": list(importance.keys()), "gain": list(importance.values())})
            imp_df.to_csv(self.feature_importance_output, index=False)
            print(f"📄 Saved feature importance → {self.feature_importance_output}")
        except Exception as e:
            print(f"  [WARN] Could not get feature importance: {e}")

    # ---------------------- STAGE 3 --------------------------
    def load_root_in_chunks_to_cudf(self, file_list, step=100_000):
        """
        Iterate over ROOT files (one or many) and yield cuDF DataFrames containing the requested features.
        Uses uproot.iterate with numpy backend then converts arrays to cuDF (cupy-backed).
        """
        tree_name = self.tree_name_stage3
        # build list of "file:treename" patterns
        targets = [f"{fn}:{tree_name}" for fn in file_list]
        for arrays in uproot.iterate(targets, self.features, step_size=step, library="np"):
            # arrays is a dict of numpy arrays
            if not arrays:
                yield cudf.DataFrame()
                continue
            # convert to cupy and then cuDF
            gdf = cudf.DataFrame({k: cp.asarray(v) for k, v in arrays.items()})
            yield gdf

    def evaluate_file(self, file_path, model):
        """
        Evaluate a single ROOT file in chunks and return numpy array of predictions.
        The model should be an xgb.Booster already loaded with GPU predictor.
        """
        preds_list = []
        for gdf in tqdm(self.load_root_in_chunks_to_cudf([file_path], step=self.eval_chunk),
                        desc=f"Eval {os.path.basename(file_path)}"):
            if gdf is None or len(gdf) == 0:
                continue

            # Try to create DMatrix directly from cuDF
            try:
                dmat = xgb.DMatrix(gdf[self.features])
            except Exception:
                # fallback: convert to cupy 2D array
                col_arrays = [gdf[col].values for col in self.features]  # these are cupy arrays
                cupy_matrix = cp.stack(col_arrays, axis=1)
                dmat = xgb.DMatrix(cupy_matrix)

            # predict with GPU predictor
            preds = model.predict(dmat)
            # preds may be numpy or cupy depending on xgboost version/build
            if isinstance(preds, cp.ndarray):
                preds = cp.asnumpy(preds)
            preds_list.append(preds)

        if preds_list:
            return np.concatenate(preds_list)
        else:
            return np.array([])

    def get_num_events(self, file_path):
        with uproot.open(file_path) as f:
            return f[self.tree_name_stage3].num_entries

    def base_process_name(self, filename):
        name = os.path.basename(filename)
        name = name.replace(".root", "")
        # Remove trailing _chunkNN or trailing numeric shard suffixes like _1, _12
        name = re.sub(r"_(chunk\d+|\d+)$", "", name)
        return name + ".root"

    def stage3_evaluate(self):
        print("\n==== Stage 3: Evaluating Model (GPU) ====")

        # Load model
        model = xgb.Booster()
        model.load_model(self.model_output)
        print(f"Loaded model: {self.model_output}")

        file_list_sig = self.signal_files_stage3
        file_list_bg = self.bg_files_stage3

        def collect(files):
            results = {}
            for file_path in files:
                print(f"  Evaluating: {file_path}")
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

        # Build output dataframe(s)
        out_frames = []
        for name, info in {**sig_res, **bg_res}.items():
            # some processes may have zero preds (e.g., empty files) so guard
            preds_chunks = [p for p in info["preds"] if p is not None and len(p) > 0]
            if preds_chunks:
                preds = np.concatenate(preds_chunks)
            else:
                preds = np.array([])

            n = len(preds)
            if n == 0:
                # still include process with zero rows? skip
                continue

            # create cuDF to hold results on GPU, then write parquet
            gdf = cudf.DataFrame({
                "process": cudf.Series([name] * n, dtype="object"),
                "event_idx": cp.arange(n),
                "bdt_score": cp.asarray(preds)
            })
            out_frames.append(gdf)

        if not out_frames:
            print("[!] No predictions produced.")
            return

        final_gdf = cudf.concat(out_frames, ignore_index=True)
        # Save as parquet (keeps data on GPU during write if supported)
        out_path = Path(self.output_unbinned)
        final_gdf.to_parquet(out_path)
        print(f"📦 Saved unbinned predictions → {out_path}")
        print(f"Shape (rows): {len(final_gdf):,}")



# ------------------------------- Example usage --------------------------------
if __name__ == "__main__":
    # Example config (modify paths to match your environment)
    config = {
        # Stage 1
        "input_dir_stage1": "./root_files/",
        "output_dir_stage1": "./pkl_stage1_parquet/",
        "tree_name_stage1": "events",
        "max_events_stage1": 100_000,
        "stage1_file_ext": ".parquet",

        # Stage 2
        # After stage1, these should be parquet files created by stage1_convert_root_to_parquet
        "train_signal": ["pkl_stage1_parquet/signal1.parquet"],
        "train_background": ["pkl_stage1_parquet/bg1.parquet", "pkl_stage1_parquet/bg2.parquet"],
        "model_output": "trained_model_gpu.json",
        "feature_importance_output": "feature_importance_gpu.csv",
        "features": ["Missing_Pt", "Jet1_Pt", "Jet2_Pt"],

        # Stage 3
        "signal_files_stage3": ["/eos/.../sig.root"],
        "bg_files_stage3": ["/eos/.../bg1.root", "/eos/.../bg2.root"],
        "output_unbinned": "bdt_scores_unbinned.parquet",
        "eval_chunk": 100_000,
        "tree_name_stage3": "events",
    }

    pipe = BDTYukawaPipelineGPU(config)

    # Run stages as before:
    # 1) convert ROOT -> GPU parquet
    # pipe.stage1_convert_root_to_parquet()

    # 2) train on GPU
    # pipe.stage2_train_bdt(num_rounds=200)

    # 3) evaluate files and save unbinned predictions
    # pipe.stage3_evaluate()

    print("Example pipeline defined. Uncomment desired stage calls in __main__ to run.")
