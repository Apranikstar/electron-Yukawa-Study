import uproot
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================
# Tau sample
# ==============================
files_C = [
    "wzp6_ee_Htaunutauqq_ecm125.root",
    "wzp6_ee_Hqqtaunutau_ecm125.root"
]

# Physics labels
file_labels = {
    "wzp6_ee_Htaunutauqq_ecm125.root": r"$H \to W(\tau \nu_\tau)\, W^*(jj)$",
    "wzp6_ee_Hqqtaunutau_ecm125.root": r"$H \to W(jj)\, W^*(\tau \nu_\tau)$"
}

branch = "Jets_InMa"
branch_label = r"$m_{jj}$ [GeV]"

# ==============================
# Prepare output folder
# ==============================
os.makedirs("figs", exist_ok=True)

# ==============================
# Helper: load a branch into NumPy
# ==============================
def load_data(file, branch):
    with uproot.open(file) as f:
        tree = f[f.keys()[0]]
        return tree[branch].array(library="np")

# ==============================
# Optimization routine
# ==============================
def optimize_cut(file1, file2, branch, n_steps=200, save_dir="figs", show=True):
    data1 = load_data(file1, branch)
    data2 = load_data(file2, branch)

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

    # Fractions
    below1 = np.sum(data1 < best_cut)
    above1 = np.sum(data1 >= best_cut)
    below2 = np.sum(data2 < best_cut)
    above2 = np.sum(data2 >= best_cut)

    print(f"\n🔹 Optimal cut for {branch}: {best_cut:.3f} GeV (max separation = {best_sep:.3f})\n")
    print(f"{file1}: below = {below1/len(data1)*100:.2f}%, above = {above1/len(data1)*100:.2f}%")
    print(f"{file2}: below = {below2/len(data2)*100:.2f}%, above = {above2/len(data2)*100:.2f}%")

    # ==============================
    # Plot separation curve
    # ==============================
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(cuts, separations, label=r"$|F_1(<cut) - F_2(<cut)|$")
    plt.axvline(best_cut, color="red", linestyle="--",
                label=f"Best cut = {best_cut:.2f} GeV\nSeparation = {best_sep:.2f}")
    plt.xlabel(branch_label)
    plt.ylabel("Separation")
    plt.title(f"Cut Optimization for {branch_label}")
    plt.legend()
    plt.xlim(0,120)
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    sep_path = f"{save_dir}/separation_{branch}_{os.path.basename(file1).replace('.root','')}_vs_{os.path.basename(file2).replace('.root','')}.png"
    plt.savefig(sep_path)
    print(f"Saved separation plot: {sep_path}")
    if show:
        plt.show()
    else:
        plt.close()

    # ==============================
    # Plot distributions
    # ==============================
    plt.figure(figsize=(8, 6), dpi=300)
    plt.hist(data1, bins=60, histtype="step", density=True,
             label=file_labels[file1], linewidth=1.5)
    plt.hist(data2, bins=60, histtype="step", density=True,
             label=file_labels[file2], linewidth=1.5)
    plt.axvline(best_cut, color="red", linestyle="--", linewidth=2,
                label=f"Optimal cut = {best_cut:.2f} GeV\nSeparation = {best_sep:.2f}")
    plt.xlabel(branch_label)
    plt.ylabel("Normalized entries")
    plt.title(f"{branch_label} distributions with optimal cut")
    plt.legend()
    plt.xlim(0,120)
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    dist_path = f"{save_dir}/distribution_{branch}_{os.path.basename(file1).replace('.root','')}_vs_{os.path.basename(file2).replace('.root','')}.png"
    plt.savefig(dist_path)
    print(f"Saved distribution plot: {dist_path}")
    if show:
        plt.show()
    else:
        plt.close()

    return best_cut, best_sep


# ==============================
# Run optimization
# ==============================
cut_Jets, sep_Jets = optimize_cut(files_C[0], files_C[1], branch, show=True)

print("\n===== Optimization Summary =====")
print(f"Jets_InMa : best cut = {cut_Jets:.2f} GeV, separation = {sep_Jets:.3f}")
