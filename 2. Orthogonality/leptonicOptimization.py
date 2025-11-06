import uproot
import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory for saving figures
os.makedirs("figs", exist_ok=True)

# File groups
files_A = [
    "wzp6_ee_Henueqq_ecm125.root",
    "wzp6_ee_Hqqenue_ecm125.root"
]

files_B = [
    "wzp6_ee_Hmunumuqq_ecm125.root",
    "wzp6_ee_Hqqmunumu_ecm125.root"
]

file_labels = {
    "wzp6_ee_Henueqq_ecm125.root":  r"$H \to W(e \nu_e)\, W^*(jj)$",
    "wzp6_ee_Hqqenue_ecm125.root":  r"$H \to W(jj)\, W^*(e \nu_e)$",
    "wzp6_ee_Hmunumuqq_ecm125.root": r"$H \to W(\mu \nu_\mu)\, W^*(jj)$",
    "wzp6_ee_Hqqmunumu_ecm125.root": r"$H \to W(jj)\, W^*(\mu \nu_\mu)$"
}

def load_data(file, branch):
    with uproot.open(file) as f:
        tree = f[f.keys()[0]]
        return tree[branch].array(library="np")

def optimize_cut(file1, file2, branch, n_steps=200, save_dir="figs", show=True):
    data1 = load_data(file1, branch)
    data2 = load_data(file2, branch)

    all_data = np.concatenate([data1, data2])
    xmin, xmax = np.percentile(all_data, [1, 99])  # cut tails for stability
    cuts = np.linspace(xmin, xmax, n_steps)

    separations = []

    for cut in cuts:
        below1 = np.sum(data1 < cut) / len(data1)
        below2 = np.sum(data2 < cut) / len(data2)
        separations.append(abs(below1 - below2))

    best_idx = np.argmax(separations)
    best_cut = cuts[best_idx]
    best_sep = separations[best_idx]

    print(f"\nOptimal cut for {branch}: {best_cut:.3f}")
    print(f"Max separation (efficiency): {best_sep:.3f}")

    # === Plot separation curve ===
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(cuts, separations, label=f"|F₁(<cut) - F₂(<cut)|")
    plt.axvline(best_cut, color="red", linestyle="--",
                label=f"Best cut = {best_cut:.2f}\nSeparation = {best_sep:.2f}")
    plt.xlabel(f"{branch}")
    plt.ylabel("Separation")
    plt.title(f"Cut optimization for {branch}")
    plt.legend()
    plt.xlim(0, 120)
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    fig_name_sep = f"{save_dir}/separation_{branch}_{os.path.basename(file1).replace('.root','')}_vs_{os.path.basename(file2).replace('.root','')}.png"
    plt.savefig(fig_name_sep)
    print(f"Saved separation plot: {fig_name_sep}")
    if show:
        plt.show()
    else:
        plt.close()

    # === Plot distributions ===
    plt.figure(figsize=(8, 6), dpi=300)
    plt.hist(data1, bins=60, histtype="step", density=True,
             label=f"{file_labels[file1]}", linewidth=1.5)
    plt.hist(data2, bins=60, histtype="step", density=True,
             label=f"{file_labels[file2]}", linewidth=1.5)
    plt.axvline(best_cut, color="red", linestyle="--", linewidth=2,
                label=f"Optimal cut = {best_cut:.2f}\nSeparation = {best_sep:.2f}")
    plt.xlabel(branch)
    plt.ylabel("Normalized entries")
    plt.legend()
    plt.title(f"{branch} distributions with optimal cut")
    plt.grid(True)
    plt.xlim(0, 120)
    plt.tight_layout()

    # Save figure
    fig_name_dist = f"{save_dir}/distributions_{branch}_{os.path.basename(file1).replace('.root','')}_vs_{os.path.basename(file2).replace('.root','')}.png"
    plt.savefig(fig_name_dist)
    print(f"Saved distribution plot: {fig_name_dist}")
    if show:
        plt.show()
    else:
        plt.close()

    return best_cut, best_sep


# Example usage:
cut_A, sep_A = optimize_cut(files_A[0], files_A[1], "Jets_InMa", show=True)
cut_B, sep_B = optimize_cut(files_B[0], files_B[1], "Jets_InMa", show=True)


##############################

import uproot
import numpy as np
import matplotlib.pyplot as plt
import os

# Create folder for saved figures
os.makedirs("figs", exist_ok=True)

# Define your file groups
files_A = [
    "wzp6_ee_Henueqq_ecm125.root",
    "wzp6_ee_Hqqenue_ecm125.root"
]

files_B = [
    "wzp6_ee_Hmunumuqq_ecm125.root",
    "wzp6_ee_Hqqmunumu_ecm125.root"
]

# Physics labels for legend
file_labels = {
    "wzp6_ee_Henueqq_ecm125.root":  r"$H \to W(e \nu_e)\, W^*(jj)$",
    "wzp6_ee_Hqqenue_ecm125.root":  r"$H \to W(jj)\, W^*(e \nu_e)$",
    "wzp6_ee_Hmunumuqq_ecm125.root": r"$H \to W(\mu \nu_\mu)\, W^*(jj)$",
    "wzp6_ee_Hqqmunumu_ecm125.root": r"$H \to W(jj)\, W^*(\mu \nu_\mu)$"
}

# Pretty axis labels
branch_labels = {
    "EnuM":  r"$m_{e\nu_e}$ [GeV]",
    "MunuM": r"$m_{\mu\nu_\mu}$ [GeV]"
}

# Helper: load a branch into NumPy
def load_data(file, branch):
    with uproot.open(file) as f:
        tree = f[f.keys()[0]]
        return tree[branch].array(library="np")

# Optimization routine
def optimize_cut(file1, file2, branch, n_steps=200, save_dir="figs", show=True):
    data1 = load_data(file1, branch)
    data2 = load_data(file2, branch)

    # Remove extreme tails for robustness
    all_data = np.concatenate([data1, data2])
    xmin, xmax = np.percentile(all_data, [1, 99])
    cuts = np.linspace(xmin, xmax, n_steps)

    separations = []
    for cut in cuts:
        below1 = np.sum(data1 < cut) / len(data1)
        below2 = np.sum(data2 < cut) / len(data2)
        separations.append(abs(below1 - below2))

    best_idx = np.argmax(separations)
    best_cut = cuts[best_idx]
    best_sep = separations[best_idx]

    # Compute fractions for the best cut
    below1 = np.sum(data1 < best_cut)
    above1 = np.sum(data1 >= best_cut)
    below2 = np.sum(data2 < best_cut)
    above2 = np.sum(data2 >= best_cut)

    print(f"\n🔹 Optimal cut for {branch}: {best_cut:.3f} (max separation = {best_sep:.3f})\n")
    print(f"{file1}: below = {below1/len(data1)*100:.2f}%, above = {above1/len(data1)*100:.2f}%")
    print(f"{file2}: below = {below2/len(data2)*100:.2f}%, above = {above2/len(data2)*100:.2f}%")

    # --- Plot separation curve ---
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(cuts, separations, label=r"$|F_1(<cut) - F_2(<cut)|$")
    plt.axvline(best_cut, color="red", linestyle="--",
                label=f"Best cut = {best_cut:.2f} GeV\nSeparation = {best_sep:.2f}")
    plt.xlabel(branch_labels.get(branch, branch))
    plt.ylabel("Separation")
    plt.title(f"Cut Optimization for {branch_labels.get(branch, branch)}")
    plt.legend()
    plt.xlim(0,120)
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    fig_sep = f"{save_dir}/separation_{branch}_{os.path.basename(file1).replace('.root','')}_vs_{os.path.basename(file2).replace('.root','')}.png"
    plt.savefig(fig_sep)
    print(f"Saved separation plot: {fig_sep}")
    if show:
        plt.show()
    else:
        plt.close()

    # --- Plot distributions ---
    plt.figure(figsize=(8, 6), dpi=300)
    plt.hist(data1, bins=60, histtype="step", density=True,
             label=file_labels[file1], linewidth=1.5)
    plt.hist(data2, bins=60, histtype="step", density=True,
             label=file_labels[file2], linewidth=1.5)
    plt.axvline(best_cut, color="red", linestyle="--", linewidth=2,
                label=f"Optimal cut = {best_cut:.2f} GeV\nSeparation = {best_sep:.2f}")
    plt.xlabel(branch_labels.get(branch, branch))
    plt.ylabel("Normalized entries")
    plt.title(f"{branch_labels.get(branch, branch)} distributions with optimal cut")
    plt.legend()
    plt.xlim(0,120)
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    fig_dist = f"{save_dir}/distributions_{branch}_{os.path.basename(file1).replace('.root','')}_vs_{os.path.basename(file2).replace('.root','')}.png"
    plt.savefig(fig_dist)
    print(f"Saved distribution plot: {fig_dist}")
    if show:
        plt.show()
    else:
        plt.close()

    return best_cut, best_sep

# --- Run the optimization ---
cut_EnuM, sep_EnuM = optimize_cut(files_A[0], files_A[1], "EnuM", show=True)
cut_MunuM, sep_MunuM = optimize_cut(files_B[0], files_B[1], "MunuM", show=True)
