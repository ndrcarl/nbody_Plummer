import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  5_plot_mass_vcirc.py
#  Verifies the Cumulative Mass and Circular Velocity equations
#  exactly as written on Page 10 of the notes.
# ============================================================

N = 10000
M_tot = 1.0
b = 1.0
G = 1.0
mass_per_particle = M_tot / N

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

# Center the positions
cm = np.mean(pos, axis=0)
r = np.linalg.norm(pos - cm, axis=1)

# Sort radii to compute cumulative mass
r_sorted = np.sort(r)
# Cumulative mass: particle 1 has mass m, particle 2 has 2m, etc.
M_sim = np.arange(1, N + 1) * mass_per_particle

# Avoid division by zero at r=0
mask = r_sorted > 0.01
r_clean = r_sorted[mask]
M_clean = M_sim[mask]

# Simulated Circular Velocity: v_circ = sqrt(G * M(<r) / r)
v_circ_sim = np.sqrt(G * M_clean / r_clean)

# Analytical Formulas from Page 10 of the notes
r_theory = np.logspace(-2, 1.5, 500)
M_theory = M_tot * (r_theory**3) / (b**3 * (1.0 + (r_theory/b)**2)**1.5)
v_circ_theory = np.sqrt(G * M_theory / r_theory)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Plummer Sphere: Mass Profile & Circular Velocity (Notes Page 10)', fontsize=14)

# Left Panel: Cumulative Mass
axes[0].plot(r_theory, M_theory, color='crimson', lw=2.5, label='Theory $M(r)$')
axes[0].plot(r_clean, M_clean, color='steelblue', lw=2, linestyle='--', label='N-body Simulation')
axes[0].axhline(M_tot, color='k', ls=':', label='Total Mass')
axes[0].set_xlabel('Radius $r$ [code units]')
axes[0].set_ylabel('Cumulative Mass $M(<r)$')
axes[0].set_title('Cumulative Mass Profile')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right Panel: Circular Velocity
axes[1].plot(r_theory, v_circ_theory, color='crimson', lw=2.5, label='Theory $v_{circ}(r)$')
axes[1].plot(r_clean, v_circ_sim, color='steelblue', lw=2, linestyle='--', label='N-body Simulation')
axes[1].set_xlabel('Radius $r$[code units]')
axes[1].set_ylabel('Circular Velocity $v_{circ}$')
axes[1].set_title('Rotation Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plummer_mass_vcirc.pdf", dpi=200)
print("Saved plummer_mass_vcirc.pdf")
