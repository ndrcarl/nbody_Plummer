import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  virial_stability.py
#  Calculates the Virial Ratio 2K/|W| over time.
#  A system in equilibrium must stay at 1.0.
# ============================================================

N = 10000
MASS = 0.0001

def read_energies(filepath):
    times, virial_ratio = [],[]
    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            try:
                if int(line.strip()) != N: continue
            except ValueError: continue

            f.readline() # NDIM
            t = float(f.readline().strip())
            for _ in range(N): f.readline() # mass
            for _ in range(N): f.readline() # pos
            
            vel = np.empty((N, 3))
            for j in range(N): vel[j] = np.fromstring(f.readline(), sep=' ')
            
            phi = np.empty(N)
            for j in range(N): phi[j] = float(f.readline())

            # K = 1/2 sum(m v^2)
            K = 0.5 * MASS * np.sum(vel**2)
            # W = 1/2 sum(m phi)
            W = 0.5 * MASS * np.sum(phi)
            
            times.append(t)
            virial_ratio.append(2.0 * K / abs(W))
            
    return np.array(times), np.array(virial_ratio)

times, vr = read_energies("plummer.out")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(times, vr, color='purple', lw=2, label='$2K/|W|$')
ax.axhline(1.0, color='k', ls='--', lw=1.5, label='Virial Equilibrium (1.0)')

ax.set_ylim(0.8, 1.2)
ax.set_xlabel('Time [code units]')
ax.set_ylabel('Virial Ratio $2K/|W|$')
ax.set_title('Plummer Sphere Stability: Virial Ratio')
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("virial_ratio.pdf", dpi=200)
print(f"Saved virial_ratio.pdf using data from {filename}")
