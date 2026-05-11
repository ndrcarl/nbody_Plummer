import numpy as np
import matplotlib.pyplot as plt
import sys

N = 10000
TARGET_TIMES = [0.0, 5.0, 10.0, 20.0]

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
                    snaps[tt] = pos.copy()
    return snaps

data = load_times(sys.argv[1])

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.suptitle('Plummer sphere snapshots (x-y projection, all %d particles)' % N)

for i, t in enumerate(TARGET_TIMES):
    ax = axes[i]
    if t not in data:
        ax.set_title('t = %.1f  (not found)' % t)
        continue
    p = data[t]
    ax.scatter(p[:, 0], p[:, 1], s=1, color='k', alpha=0.15)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title('t = %.1f' % t)
    ax.set_xlabel('x')
    if i == 0:
        ax.set_ylabel('y')

plt.tight_layout()
plt.savefig("snapshots_2d.pdf")
print("Saved snapshots_2d.pdf")
