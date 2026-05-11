import numpy as np
import matplotlib.pyplot as plt
import sys

N = 10000

filename = sys.argv[1] if len(sys.argv) > 1 else "plummer.out"

tracked_ids = None
trajectories = {0: {'x': [], 'y': []},
                1: {'x': [], 'y': []},
                2: {'x': [], 'y': []}}

with open(filename, 'r') as f:
    while True:
        line = f.readline()
        if not line: break
        try:
            if int(line.strip()) != N: continue
        except: continue
        f.readline()
        t = float(f.readline().strip())
        for _ in range(N): f.readline()
        pos = np.empty((N, 3))
        for j in range(N): pos[j] = np.fromstring(f.readline(), sep=' ')
        for _ in range(N): f.readline()
        for _ in range(N): f.readline()

        if tracked_ids is None:
            r = np.linalg.norm(pos, axis=1)
            id_core = np.argmin(np.abs(r - 0.3))   # inside core
            id_mid  = np.argmin(np.abs(r - 1.3))   # near r_hm
            id_halo = np.argmin(np.abs(r - 3.5))   # halo
            tracked_ids = [id_core, id_mid, id_halo]

        for i, p_id in enumerate(tracked_ids):
            trajectories[i]['x'].append(pos[p_id, 0])
            trajectories[i]['y'].append(pos[p_id, 1])

colors = ['blue', 'red', 'green']
labels = ['core  r~0.3', 'half-mass  r~1.3', 'halo  r~3.5']

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Orbital rosettes in Plummer potential (Bertrand's theorem)")

for i in range(3):
    axes[i].plot(trajectories[i]['x'], trajectories[i]['y'], color=colors[i], lw=0.8)
    axes[i].plot(0, 0, 'k+', ms=10)
    axes[i].set_aspect('equal')
    axes[i].set_title(labels[i])
    axes[i].set_xlabel('x')
    axes[i].grid(True)

axes[0].set_ylabel('y')
plt.tight_layout()
plt.savefig("plummer_orbits.pdf")
print("Saved plummer_orbits.pdf")
