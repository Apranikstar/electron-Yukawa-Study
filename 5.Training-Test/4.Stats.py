import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

# ------------------ INPUT ------------------
# Load your unbinned DataFrame
input_pkl = "bdt_distributions/bdt_scores_unbinned_5class.pkl"
df = pd.read_pickle(input_pkl)
print(f"✅ Loaded {len(df)} events from {input_pkl}")
print(f"   Processes found: {df['process'].nunique()}")
print(f"   Total events: {len(df):,}\n")

# Define initial cross sections and generated events per sample
samples_info = {
    "wzp6_ee_Henueqq_ecm125.root": {"cross_section_init": 0.43e-05, "n_gen": 900000},
    "wzp6_ee_Hqqenue_ecm125.root": {"cross_section_init": 0.43e-05, "n_gen": 1000000},

    "wzp6_ee_Hmunumuqq_ecm125.root": {"cross_section_init": 0.43e-05, "n_gen": 1000000},
    "wzp6_ee_Hqqmunumu_ecm125.root": {"cross_section_init": 0.43e-05, "n_gen": 1000000},

    "wzp6_ee_Htaunutauqq_ecm125.root": {"cross_section_init": 0.43e-05, "n_gen": 1000000},
    "wzp6_ee_Hqqtaunutau_ecm125.root": {"cross_section_init": 0.43e-05, "n_gen": 1000000},

    "wzp6_ee_enueqq_ecm125.root": {"cross_section_init": 2.613e-02, "n_gen": 99600000},
    "wzp6_ee_eeqq_ecm125.root": {"cross_section_init": 3.934, "n_gen": 99800000},

    "wzp6_ee_munumuqq_ecm125.root": {"cross_section_init": 6.711e-03, "n_gen": 99800000},
    "wzp6_ee_mumuqq_ecm125.root": {"cross_section_init": 1.505e-1, "n_gen": 100000000},

    "wzp6_ee_taunutauqq_ecm125.root": {"cross_section_init": 6.761e-03, "n_gen": 99400000},
    "wzp6_ee_tautauqq_ecm125.root": {"cross_section_init": 1.476e-1, "n_gen": 98965252},
    "p8_ee_ZZ_4tau_ecm125.root": {"cross_section_init": 0.003, "n_gen": 100000000},

    "wzp6_ee_Htautau_ecm125.root": {"cross_section_init": 1.011e-4, "n_gen": 10000000},
    "wzp6_ee_Hllnunu_ecm125.root": {"cross_section_init": 3.187e-05, "n_gen": 1200000},

    "wzp6_ee_eenunu_ecm125.root": {"cross_section_init": 6.574e-01, "n_gen": 100000000},
    "wzp6_ee_mumununu_ecm125.root": {"cross_section_init": 2.202e-01, "n_gen": 99400000},
    "wzp6_ee_tautaununu_ecm125.root": {"cross_section_init": 4.265e-02, "n_gen": 99900000},
    "wzp6_ee_l1l2nunu_ecm125.root": {"cross_section_init": 9.845e-03, "n_gen": 99500000},
    "wzp6_ee_tautau_ecm125.root": {"cross_section_init": 25.939, "n_gen": 10000000},

    "wzp6_ee_Hgg_ecm125.root": {"cross_section_init": 7.384e-05, "n_gen": 1200000},
    "wzp6_ee_Hbb_ecm125.root": {"cross_section_init": 1.685e-3, "n_gen": 9900000},

    "wzp6_ee_qq_ecm125.root": {"cross_section_init": 3.631e+02, "n_gen": 498704128},
}

# Signal process names
signal_processes = ["wzp6_ee_Henueqq_ecm125.root", "wzp6_ee_Htaunutauqq_ecm125.root"]

# Class labels (must match training)
class_labels = ["signal", "ww", "zz_leptonic", "zz_2lep2j", "zstar"]

# ------------------ USER INPUT ------------------
print("="*60)
print("SIGNIFICANCE-OPTIMIZED 5D BINNING")
print("="*60)
luminosity = float(input("\nEnter luminosity (in 1/pb): "))

use_systematics_input = input("Include systematic uncertainties for Profile Likelihood? (y/n): ").lower()
use_systematics = use_systematics_input == 'y'
if use_systematics:
    sys_uncertainty = float(input("Enter systematic uncertainty (e.g., 0.05 for 5%): "))
else:
    sys_uncertainty = 0.05

min_signal_per_bin = float(input("\nEnter minimum signal events per bin (e.g., 1.0): "))
n_bins_range = input("Enter bin range to test (e.g., '5-20' or '10' for fixed): ")

# Parse bin range
if '-' in n_bins_range:
    bin_min, bin_max = map(int, n_bins_range.split('-'))
    bins_to_test = range(bin_min, bin_max + 1)
else:
    bins_to_test = [int(n_bins_range)]

# Create output directory
import os
output_dir = "binnedData"
os.makedirs(output_dir, exist_ok=True)

# ------------------ SIGNIFICANCE CALCULATORS ------------------

def calculate_simple_significance(signal, background):
    """Z = S/√B"""
    if background <= 0:
        return 0.0
    return signal / np.sqrt(background)

def calculate_asimov_significance(signal, background):
    """Asimov formula: Z = √(2((S+B)ln(1+S/B) - S))"""
    if signal <= 0 or background <= 0:
        return 0.0
    s, b = signal, background
    return np.sqrt(2 * ((s + b) * np.log(1 + s/b) - s))

def calculate_profile_likelihood_significance(signal_bins, background_bins, use_sys=False, sys_unc=0.05):
    """
    Profile likelihood ratio test statistic
    Tests hypothesis: mu=0 (background only) vs mu=1 (signal+background)
    """
    n_bins = len(signal_bins)
    
    def negative_log_likelihood(mu, nuisance=None):
        """Negative log-likelihood for signal strength mu"""
        nll = 0.0
        for i in range(n_bins):
            s_i = signal_bins[i]
            b_i = background_bins[i]
            
            if use_sys and nuisance is not None:
                # Include systematic uncertainty as nuisance parameter
                b_effective = b_i * (1 + nuisance[i])
                expected = mu * s_i + b_effective
                
                # Gaussian constraint on nuisance parameter
                nll += 0.5 * (nuisance[i] / sys_unc) ** 2
            else:
                expected = mu * s_i + b_i
            
            # Poisson likelihood for observed = expected (Asimov dataset)
            if expected > 0:
                observed = expected  # Asimov: set observed = expected
                nll -= poisson.logpmf(int(observed), expected)
        
        return nll
    
    # Fit background-only hypothesis (mu=0)
    if use_sys:
        # Minimize over nuisance parameters
        def nll_bkg_only(nuisance):
            return negative_log_likelihood(0.0, nuisance)
        
        initial_nuisance = np.zeros(n_bins)
        result_bkg = minimize(nll_bkg_only, initial_nuisance, method='L-BFGS-B')
        nll_0 = result_bkg.fun
    else:
        nll_0 = negative_log_likelihood(0.0)
    
    # Fit signal+background hypothesis (mu=1, profiled)
    if use_sys:
        def nll_sig_bkg(params):
            # params = [mu, nuisance_0, nuisance_1, ...]
            mu = params[0]
            nuisance = params[1:]
            return negative_log_likelihood(mu, nuisance)
        
        initial_params = np.concatenate([[1.0], np.zeros(n_bins)])
        bounds = [(0, None)] + [(-3, 3)] * n_bins  # Constrain nuisance to ±3σ
        result_sig = minimize(nll_sig_bkg, initial_params, method='L-BFGS-B', bounds=bounds)
        nll_1 = result_sig.fun
    else:
        nll_1 = negative_log_likelihood(1.0)
    
    # Test statistic: q_0 = -2 * (nll_0 - nll_1)
    q_0 = 2 * (nll_0 - nll_1)
    
    # Significance: Z = √q_0 (asymptotic approximation)
    significance = np.sqrt(max(0, q_0))
    
    return significance

def calculate_total_significance(signal_per_bin, background_per_bin, method, use_sys=False, sys_unc=0.05):
    """Calculate total significance across all bins"""
    if method == 1:
        # Simple: combine bins with √(Σ(S²/B))
        z_squared = sum((s**2 / b if b > 0 else 0) for s, b in zip(signal_per_bin, background_per_bin))
        return np.sqrt(z_squared)
    elif method == 2:
        # Asimov: sum signal and background, then calculate
        total_s = sum(signal_per_bin)
        total_b = sum(background_per_bin)
        return calculate_asimov_significance(total_s, total_b)
    elif method == 3:
        # Profile likelihood with all bins
        return calculate_profile_likelihood_significance(
            signal_per_bin, background_per_bin, use_sys, sys_unc
        )

# ------------------ COMPUTE WEIGHTS ------------------
print("\n" + "="*60)
print("STEP 1: Computing event weights")
print("="*60)

df['weight'] = 0.0
df['is_signal'] = False

for process, group_df in df.groupby('process'):
    info = samples_info.get(process)
    if info is None:
        print(f"⚠️  Skipping {process}: no cross section info found.")
        continue
    
    n_total = group_df['total_events_in_file'].iloc[0]
    sigma_init = info['cross_section_init']
    n_gen = info['n_gen']
    
    efficiency = n_total / n_gen
    sigma_final = sigma_init * efficiency
    weight = (sigma_final * luminosity) / n_total
    
    df.loc[group_df.index, 'weight'] = weight
    df.loc[group_df.index, 'is_signal'] = process in signal_processes
    
    print(f"✓ {process}: σ_final = {sigma_final:.3e} pb, weight = {weight:.3e}")

total_signal_events = df[df['is_signal']]['weight'].sum()
total_background_events = df[~df['is_signal']]['weight'].sum()
print(f"\n📊 Total signal events: {total_signal_events:.2f}")
print(f"📊 Total background events: {total_background_events:.2f}")

# ------------------ OPTIMIZE BINNING FOR ALL METHODS ------------------
print("\n" + "="*60)
print("STEP 2: Optimizing binning for all significance methods")
print("="*60)

signal_df = df[df['is_signal']].copy()
all_bdt = df[['bdt_signal', 'bdt_ww', 'bdt_zz_leptonic', 'bdt_zz_2lep2j', 'bdt_zstar']].values
signal_bdt = signal_df[['bdt_signal', 'bdt_ww', 'bdt_zz_leptonic', 'bdt_zz_2lep2j', 'bdt_zstar']].values

# Sample for clustering if needed
if len(signal_bdt) > 50000:
    print(f"   Sampling {50000} events for clustering")
    sample_idx = np.random.choice(len(signal_bdt), 50000, replace=False)
    signal_bdt_sample = signal_bdt[sample_idx]
else:
    signal_bdt_sample = signal_bdt

# Perform hierarchical clustering once
print("🔄 Performing hierarchical clustering...")
Z = linkage(signal_bdt_sample, method='ward')

# Store results for all methods
method_results = {}
significance_methods = {
    1: "Simple (S/√B)",
    2: "Asimov",
    3: "Profile Likelihood"
}

for method_id, method_name in significance_methods.items():
    print(f"\n{'='*60}")
    print(f"METHOD {method_id}: {method_name}")
    print(f"{'='*60}")
    
    best_significance = -1
    best_n_bins = None
    best_bin_assignments = None
    significance_results = []
    
    print(f"🔍 Testing {len(bins_to_test)} different bin configurations...\n")
    
    for n_bins in bins_to_test:
        # Cut tree to get n_bins clusters
        cluster_labels_sample = fcluster(Z, n_bins, criterion='maxclust')
        
        # Compute cluster centers
        unique_labels = np.unique(cluster_labels_sample)
        cluster_centers = np.array([
            signal_bdt_sample[cluster_labels_sample == label].mean(axis=0)
            for label in unique_labels
        ])
        
        # Assign all events to nearest cluster
        distances = cdist(all_bdt, cluster_centers, metric='euclidean')
        bin_assignments = np.argmin(distances, axis=1)
        
        # Calculate signal and background per bin
        df_temp = df.copy()
        df_temp['bin_id'] = bin_assignments
        
        signal_per_bin = []
        background_per_bin = []
        
        for bin_id in range(len(cluster_centers)):
            bin_events = df_temp[df_temp['bin_id'] == bin_id]
            s = bin_events[bin_events['is_signal']]['weight'].sum()
            b = bin_events[~bin_events['is_signal']]['weight'].sum()
            signal_per_bin.append(s)
            background_per_bin.append(b)
        
        # Check minimum signal constraint
        min_signal = min(signal_per_bin) if signal_per_bin else 0
        
        if min_signal < min_signal_per_bin:
            print(f"  n_bins = {n_bins:3d} → SKIPPED (min signal/bin = {min_signal:.2f} < {min_signal_per_bin})")
            continue
        
        # Calculate significance for this method
        if method_id == 3 and use_systematics:
            sig = calculate_total_significance(
                signal_per_bin, background_per_bin, 
                method_id, use_sys=True, sys_unc=sys_uncertainty
            )
        else:
            sig = calculate_total_significance(
                signal_per_bin, background_per_bin, method_id
            )
        
        significance_results.append({
            'n_bins': n_bins,
            'significance': sig,
            'min_signal_per_bin': min_signal
        })
        
        print(f"  n_bins = {n_bins:3d} → Significance = {sig:.3f}, Min signal/bin = {min_signal:.2f}")
        
        if sig > best_significance:
            best_significance = sig
            best_n_bins = n_bins
            best_bin_assignments = bin_assignments.copy()
    
    if best_n_bins is None:
        print(f"⚠️  No binning configuration meets min_signal_per_bin = {min_signal_per_bin} for this method")
        continue
    
    print(f"\n✨ OPTIMAL BINNING FOR {method_name} ✨")
    print(f"   Number of bins: {best_n_bins}")
    print(f"   Maximum significance: {best_significance:.3f}")
    
    # Store results
    method_results[method_id] = {
        'method_name': method_name,
        'best_n_bins': best_n_bins,
        'best_significance': best_significance,
        'best_bin_assignments': best_bin_assignments,
        'significance_results': significance_results
    }

# ------------------ CREATE FINAL BINNED DATAFRAMES FOR ALL METHODS ------------------
print("\n" + "="*60)
print("STEP 3: Creating binned dataframes for all methods")
print("="*60)

final_results_summary = []

for method_id, result in method_results.items():
    method_name = result['method_name']
    print(f"\nProcessing {method_name}...")
    
    df['bin_id'] = result['best_bin_assignments']
    
    results = []
    
    for (process, bin_id), group in df.groupby(['process', 'bin_id']):
        info = samples_info.get(process)
        if info is None:
            continue
        
        n_total = group['total_events_in_file'].iloc[0]
        sigma_init = info['cross_section_init']
        n_gen = info['n_gen']
        
        efficiency = n_total / n_gen
        sigma_final = sigma_init * efficiency
        
        bin_count = len(group)
        reco_events = group['weight'].sum()
        bin_fraction = bin_count / n_total
        bin_xs = sigma_final * bin_fraction
        
        # Bin center in 5D space (weighted)
        bin_center = {}
        for i, label in enumerate(class_labels):
            col = f'bdt_{label}'
            weighted_avg = np.average(group[col].values, weights=group['weight'].values)
            bin_center[f'bin_center_{label}'] = weighted_avg
        
        result_row = {
            'process': process,
            'bin_id': bin_id,
            **bin_center,
            'bin_count': bin_count,
            'bin_fraction': bin_fraction,
            'preselection_efficiency': efficiency,
            'initial_cross_section': sigma_init,
            'final_cross_section': sigma_final,
            'bin_cross_section': bin_xs,
            'luminosity': luminosity,
            'reco_level_events': reco_events
        }
        results.append(result_row)
    
    final_binned_df = pd.DataFrame(results)
    
    # Add significance per bin
    final_binned_df['signal_events'] = 0.0
    final_binned_df['background_events'] = 0.0
    
    for bin_id in final_binned_df['bin_id'].unique():
        bin_data = final_binned_df[final_binned_df['bin_id'] == bin_id]
        s = bin_data[bin_data['process'].isin(signal_processes)]['reco_level_events'].sum()
        b = bin_data[~bin_data['process'].isin(signal_processes)]['reco_level_events'].sum()
        
        final_binned_df.loc[final_binned_df['bin_id'] == bin_id, 'signal_events'] = s
        final_binned_df.loc[final_binned_df['bin_id'] == bin_id, 'background_events'] = b
    
    # Save to file
    method_label = method_name.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    output_pkl = os.path.join(output_dir, f"bdt_binned_{method_label}_5class.pkl")
    final_binned_df.to_pickle(output_pkl)
    
    print(f"  ✅ Saved → {output_pkl}")
    print(f"     Shape: {final_binned_df.shape}, Bins: {final_binned_df['bin_id'].nunique()}")
    
    # Store summary
    final_results_summary.append({
        'method_id': method_id,
        'method_name': method_name,
        'n_bins': result['best_n_bins'],
        'significance': result['best_significance'],
        'output_file': output_pkl,
        'binned_df': final_binned_df
    })

print("\n" + "="*60)
print("FINAL SUMMARY - ALL SIGNIFICANCE METHODS")
print("="*60)

print(f"\nLuminosity: {luminosity} pb⁻¹")
print(f"Total signal events: {total_signal_events:.2f}")
print(f"Total background events: {total_background_events:.2f}")
if use_systematics:
    print(f"Systematic uncertainty (Profile Likelihood): {sys_uncertainty*100:.1f}%")
print(f"Minimum signal per bin constraint: {min_signal_per_bin}")

print("\n" + "-"*60)
print(f"{'Method':<25} {'N Bins':<10} {'Significance':<15} {'Output File'}")
print("-"*60)

for summary in final_results_summary:
    method_name = summary['method_name']
    n_bins = summary['n_bins']
    sig = summary['significance']
    output_file = os.path.basename(summary['output_file'])
    print(f"{method_name:<25} {n_bins:<10} {sig:<15.3f} {output_file}")

print("-"*60)

# Print detailed per-bin breakdown for each method
for summary in final_results_summary:
    print(f"\n{'='*60}")
    print(f"DETAILED BREAKDOWN: {summary['method_name']}")
    print(f"{'='*60}")
    
    binned_df = summary['binned_df']
    method_id = summary['method_id']
    
    bin_summary = binned_df.groupby('bin_id').agg({
        'signal_events': 'first',
        'background_events': 'first'
    }).reset_index().sort_values('bin_id')
    
    print(f"\n{'Bin':<6} {'Signal':<12} {'Background':<12} {'S/B':<10} {'Z':<10}")
    print("-"*60)
    
    for _, row in bin_summary.iterrows():
        bin_id = int(row['bin_id'])
        s = row['signal_events']
        b = row['background_events']
        s_over_b = s/b if b > 0 else 0
        
        # Calculate significance for this bin
        if method_id == 1:
            z = calculate_simple_significance(s, b)
        elif method_id == 2:
            z = calculate_asimov_significance(s, b)
        else:
            z = calculate_profile_likelihood_significance([s], [b], use_systematics, sys_uncertainty)
        
        print(f"{bin_id:<6} {s:<12.2f} {b:<12.2f} {s_over_b:<10.4f} {z:<10.3f}")
    
    print(f"\nTotal significance ({summary['method_name']}): {summary['significance']:.3f}")

print("\n" + "="*60)
print("✅ ALL BINNING METHODS COMPLETE!")
print(f"✅ Output directory: {output_dir}/")
print("="*60)
