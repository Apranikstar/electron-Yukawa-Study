import uproot
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Define file groups
# ==============================
files_A = [
    "wzp6_ee_Henueqq_ecm125.root",
    "wzp6_ee_Hqqenue_ecm125.root"
]

files_B = [
    "wzp6_ee_Hmunumuqq_ecm125.root",
    "wzp6_ee_Hqqmunumu_ecm125.root"
]

files_C = [
    "wzp6_ee_Htaunutauqq_ecm125.root",
    "wzp6_ee_Hqqtaunutau_ecm125.root"
]

# ==============================
# Physics labels
# ==============================
file_labels = {
    "wzp6_ee_Henueqq_ecm125.root":   r"$H \to W(e \nu_e)\, W^*(jj)$",
    "wzp6_ee_Hqqenue_ecm125.root":   r"$H \to W(jj)\, W^*(e \nu_e)$",
    "wzp6_ee_Hmunumuqq_ecm125.root": r"$H \to W(\mu \nu_\mu)\, W^*(jj)$",
    "wzp6_ee_Hqqmunumu_ecm125.root": r"$H \to W(jj)\, W^*(\mu \nu_\mu)$",
    "wzp6_ee_Htaunutauqq_ecm125.root": r"$H \to W(\tau \nu_\tau)\, W^*(jj)$",
    "wzp6_ee_Hqqtaunutau_ecm125.root": r"$H \to W(jj)\, W^*(\tau \nu_\tau)$"
}

# ==============================
# Axis labels (LaTeX formatted)
# ==============================
branch_labels = {
    "EnuM":      r"$m_{e\nu_e}$ [GeV]",
    "MunuM":     r"$m_{\mu\nu_\mu}$ [GeV]",
    "Jets_InMa": r"$m_{jj}$ [GeV]"
}

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
def optimize_cut(file1, file2, branch, n_steps=200):
    data1 = load_data(file1, branch)
    data2 = load_data(file2, branch)

    # avoid outliers
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
    plt.figure(figsize=(8, 6))
    plt.plot(cuts, separations, label=r"$|F_1(<cut) - F_2(<cut)|$")
    plt.axvline(best_cut, color="red", linestyle="--", label=f"Best cut = {best_cut:.2f} GeV")
    plt.xlabel(branch_labels.get(branch, branch))
    plt.ylabel("Separation")
    plt.title(f"Cut Optimization for {branch_labels.get(branch, branch)}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot distributions ---
    plt.figure(figsize=(8, 6))
    plt.hist(data1, bins=60, histtype="step", density=True, label=file_labels[file1], linewidth=1.5)
    plt.hist(data2, bins=60, histtype="step", density=True, label=file_labels[file2], linewidth=1.5)
    plt.axvline(best_cut, color="red", linestyle="--", linewidth=2, label=f"Optimal cut = {best_cut:.2f} GeV")
    plt.xlabel(branch_labels.get(branch, branch))
    plt.ylabel("Normalized entries")
    plt.title(f"{branch_labels.get(branch, branch)} distributions with optimal cut")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return best_cut, best_sep

# ==============================
# Run optimization for all sets
# ==============================
cut_EnuM, sep_EnuM = optimize_cut(files_A[0], files_A[1], "EnuM")
cut_MunuM, sep_MunuM = optimize_cut(files_B[0], files_B[1], "MunuM")
cut_Jets, sep_Jets  = optimize_cut(files_C[0], files_C[1], "Jets_InMa")

print("\n===== Optimization Summary =====")
print(f"EnuM      : best cut = {cut_EnuM:.2f} GeV, separation = {sep_EnuM:.3f}")
print(f"MunuM     : best cut = {cut_MunuM:.2f} GeV, separation = {sep_MunuM:.3f}")
print(f"Jets_InMa : best cut = {cut_Jets:.2f} GeV, separation = {sep_Jets:.3f}")
