import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  4_plot_orbits.py
#  Proves Bertrand's Theorem by showing orbital precession (Rosettes).
# ============================================================

N = 10000

import sys

# Use the file passed in the terminal, OR default to run_001
filename = sys.argv[1] if len(sys.argv) > 1 else "run_001/plummer.out"

# We need to pick 3 particles at t=0 to track.
# Particle 1: Deep in the core
# Particle 2: Near half-mass radius
# Particle 3: In the halo
tracked_ids = None
trajectories = {0: {'x':[], 'y':[]}, 1: {'x':[], 'y':[]}, 2: {'x':[], 'y':[]}}
with open(filename, 'r') as f:
    while True:
        line = f.readline()
        if not line: break
        try:
            if int(line.strip()) != N: continue
        except: continue
        
        f.readline() # NDIM
        t = float(f.readline().strip())
        for _ in range(N): f.readline() # mass
        
        pos = np.empty((N, 3))
        for j in range(N): pos[j] = np.fromstring(f.readline(), sep=' ')
        
        for _ in range(N): f.readline() # vel
        for _ in range(N): f.readline() # phi
        
        # At snapshot 0, figure out which particles to track
        if tracked_ids is None:
            r = np.linalg.norm(pos, axis=1)
            id_core = np.argmin(np.abs(r - 0.3))  # Core
            id_mid  = np.argmin(np.abs(r - 1.0))  # Half-mass
            id_halo = np.argmin(np.abs(r - 2.5))  # Halo
            tracked_ids =[id_core, id_mid, id_halo]
        
        # Save their X and Y coordinates for plotting
        for i, p_id in enumerate(tracked_ids):
            trajectories[i]['x'].append(pos[p_id, 0])
            trajectories[i]['y'].append(pos[p_id, 1])

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Orbital Precession & Bertrand's Theorem (Non-Closed Orbits)", fontsize=14)

colors =['crimson', 'steelblue', 'green']
titles =['Core Orbit ($r \\approx 0.3$)', 'Half-Mass Orbit ($r \\approx 1.0$)', 'Halo Orbit ($r \\approx 2.5$)']

for i in range(3):
    ax = axes[i]
    ax.plot(trajectories[i]['x'], trajectories[i]['y'], color=colors[i], lw=1.0, alpha=0.8)
    ax.scatter(0, 0, color='black', marker='+', s=100) # Center
    ax.set_aspect('equal')
    ax.set_title(titles[i])
    ax.set_xlabel('X coordinate')
    if i == 0: ax.set_ylabel('Y coordinate')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plummer_orbits.pdf", dpi=200)
print("Saved plummer_orbits.pdf")
