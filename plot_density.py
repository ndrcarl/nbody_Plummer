import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  1_plot_density.py
#  Verifies the spatial distribution matches the m=5 Polytrope.
# ============================================================

N = 10000
M_tot = 1.0
b = 1.0

# Read the last snapshot
import sys

# Use the file passed in the terminal, OR default to run_001
filename = sys.argv[1] if len(sys.argv) > 1 else "run_001/plummer.out"

# Read the last snapshot
pos = None
with open(filename, 'r') as f:
    while True:
        line = f.readline()
        if not line: break
        try:
            if int(line.strip()) != N: continue
        except: continue
        f.readline(); f.readline()
        for _ in range(N): f.readline()
        
        pos = np.empty((N, 3))
        for j in range(N): pos[j] = np.fromstring(f.readline(), sep=' ')
        
        for _ in range(N): f.readline()
        for _ in range(N): f.readline()

# Calculate radii relative to Center of Mass
cm = np.mean(pos, axis=0)
r = np.linalg.norm(pos - cm, axis=1)

# Create logarithmic bins
bins = np.logspace(-1.5, 1.5, 30)
counts, edges = np.histogram(r, bins=bins)

# Calculate volume of each shell to get density
volumes = (4.0 / 3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)
densities = (counts * (M_tot / N)) / volumes
r_mid = np.sqrt(edges[1:] * edges[:-1])  # Geometric mean for log bins

# Analytical m=5 Polytrope (Plummer) Density
rho_analytical = (3.0 * M_tot / (4.0 * np.pi * b**3)) * (1.0 + (r_mid/b)**2)**(-2.5)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(r_mid, densities, color='steelblue', label='N-body Simulation', zorder=3)
ax.plot(r_mid, rho_analytical, color='crimson', lw=2, label='m=5 Polytrope Theory', zorder=2)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Radius $r$ [code units]')
ax.set_ylabel('Density $\\rho(r)$ [code units]')
ax.set_title('Density Profile vs. Analytical $m=5$ Polytrope')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig("plummer_density.pdf", dpi=200)
print("Saved plummer_density.pdf")
