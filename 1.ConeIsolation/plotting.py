import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("parsed_results.csv")

# Get unique drmax values
drmax_values = sorted(df['drmax'].unique())
processes = df['process'].unique()

# Map raw process names → LaTeX labels
process_labels = {
    "wzp6_ee_Henueqq_ecm125":      r"$H \to W(e \nu_e) W^* (jj)$",
    "wzp6_ee_Hqqenue_ecm125":      r"$H \to W(jj) W^* (e \nu_e)$",
    "wzp6_ee_Hmunumuqq_ecm125":    r"$H \to W(\mu \nu_\mu) W^* (jj)$",
    "wzp6_ee_Hqqmunumu_ecm125":    r"$H \to W(jj) W^* (\mu \nu_\mu)$",
    "wzp6_ee_Htaunutauqq_ecm125":  r"$H \to W(\tau \nu_\tau) W^* (jj)$",
    "wzp6_ee_Hqqtaunutau_ecm125":  r"$H \to W(jj) W^* (\tau \nu_\tau)$",
    "wzp6_ee_qq_ecm125":           r"$Z^* \to jj$",
}

# Define line styles and markers
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^', 'v', '*', 'x', 'p']

# Create subplots stacked vertically
fig, axes = plt.subplots(len(drmax_values), 1, figsize=(12, 6 * len(drmax_values)), sharex=True , dpi = 300)

# Handle case of single subplot (axes not being an array)
if len(drmax_values) == 1:
    axes = [axes]

# Plot each drmax in its own subplot
for ax, drmax in zip(axes, drmax_values):
    df_drmax = df[df['drmax'] == drmax]
    
    # Separate Z* process so it's plotted last
    non_z_processes = [p for p in processes if p != "wzp6_ee_qq_ecm125"]
    z_process = ["wzp6_ee_qq_ecm125"] if "wzp6_ee_qq_ecm125" in processes else []

    for i, process in enumerate(non_z_processes + z_process):
        df_proc = df_drmax[df_drmax['process'] == process]
        if not df_proc.empty:
            label = process_labels.get(process, process)
            ax.plot(
                df_proc['selection'], 
                df_proc['reduction_factor'],
                linestyle=line_styles[i % len(line_styles)],
                marker=markers[i % len(markers)],
                label=label,
                zorder=2 if process != "wzp6_ee_qq_ecm125" else 1  # draw Z* behind signals
            )
    
    # Add horizontal reference line
    ax.axhline(0.35, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    ax.set_title(f"drmax = {drmax}")
    ax.set_ylabel("Reduction Factor")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10)

axes[-1].set_xlabel("Selection")

plt.tight_layout()
plt.savefig("stacked.png")
plt.show()
