import numpy as np
import matplotlib.pyplot as plt
import sys

N = 10000
TARGET_TIMES =[0.0, 3.0, 6.0, 10.0]

def load_times(filepath):
    snaps = {}
    with open(filepath, 'r') as f:
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
            
            for tt in TARGET_TIMES:
                if abs(t - tt) < 0.1 and tt not in snaps:
                    snaps[tt] = pos
    return snaps

data = load_times(sys.argv[1])
fig = plt.figure(figsize=(16, 4))
for i, t in enumerate(TARGET_TIMES):
    if t in data:
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        p = data[t]
        ax.scatter(p[:,0], p[:,1], p[:,2], s=1, c='steelblue', alpha=0.2)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(-3, 3)
        ax.set_title(f"t = {t:.1f}")
plt.tight_layout()
plt.savefig("snapshots_3d.pdf", dpi=150)
