import numpy as np
import matplotlib.pyplot as plt
import os
import glob

N = 10000

run_dirs = sorted(glob.glob("run_*"))
all_r50 = []
all_vr =[]
times = None

# We reuse the parsing logic from raggio and virial_stability
for d in run_dirs:
    # Read raggio
    t_list, r50 = [],[]
    with open(os.path.join(d, "plummer.out"), 'r') as f:
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
            vel = np.empty((N, 3))
            for j in range(N): vel[j] = np.fromstring(f.readline(), sep=' ')
            phi = np.empty(N)
            for j in range(N): phi[j] = float(f.readline())
            
            cm = np.mean(pos, axis=0)
            r50.append(np.percentile(np.linalg.norm(pos - cm, axis=1), 50))
            
            K = 0.5 * 0.0001 * np.sum(vel**2)
            W = 0.5 * 0.0001 * np.sum(phi)
            all_vr.append(2.0 * K / abs(W))
            t_list.append(t)
            
    if times is None: times = np.array(t_list)
    all_r50.append(r50)

# Reshape
all_vr = np.array(all_vr).reshape(len(run_dirs), -1)
all_r50 = np.array(all_r50)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Plummer Sphere Stability — {len(run_dirs)} Realizations')

# Panel 1: R50
axes[0].fill_between(times, np.min(all_r50, axis=0), np.max(all_r50, axis=0), alpha=0.3)
axes[0].plot(times, np.median(all_r50, axis=0), color='steelblue', lw=2)
axes[0].axhline(1.305, color='k', ls='--', label='Theoretical $R_{hm}$')
axes[0].set_xlabel('Time'); axes[0].set_ylabel('$R_{50}$')
axes[0].legend()

# Panel 2: Virial
axes[1].fill_between(times, np.min(all_vr, axis=0), np.max(all_vr, axis=0), color='purple', alpha=0.3)
axes[1].plot(times, np.median(all_vr, axis=0), color='purple', lw=2)
axes[1].axhline(1.0, color='k', ls='--', label='Virial Eq (1.0)')
axes[1].set_xlabel('Time'); axes[1].set_ylabel('$2K/|W|$')
axes[1].set_ylim(0.8, 1.2)
axes[1].legend()

plt.tight_layout()
plt.savefig("summary_stability.pdf", dpi=300)
print("Saved summary_stability.pdf")
