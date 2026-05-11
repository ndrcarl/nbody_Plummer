import numpy as np
import matplotlib.pyplot as plt
import sys

N = 10000
M_tot = 1.0
b = 1.0
G = 1.0

filename = sys.argv[1] if len(sys.argv) > 1 else "plummer.out"

pos, vel = None, None
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
        vel = np.empty((N, 3))
        for j in range(N): vel[j] = np.fromstring(f.readline(), sep=' ')
        for _ in range(N): f.readline()

pos -= np.mean(pos, axis=0)
vel -= np.mean(vel, axis=0)

r = np.linalg.norm(pos, axis=1)
v_r  = np.sum(pos * vel, axis=1) / r
v_t2 = np.sum(vel**2, axis=1) - v_r**2

bins   = np.logspace(-1.0, 1.0, 20)
r_mid  = np.sqrt(bins[1:] * bins[:-1])
sig_r  = np.full(len(r_mid), np.nan)
sig_t  = np.full(len(r_mid), np.nan)

for i in range(len(bins) - 1):
    mask = (r >= bins[i]) & (r < bins[i+1])
    if mask.sum() > 5:
        sig_r[i] = np.std(v_r[mask])
        sig_t[i] = np.sqrt(np.mean(v_t2[mask]) / 2.0)

# analytical Jeans solution for isotropic Plummer sphere
sigma_theory = np.sqrt(G * M_tot / (6.0 * np.sqrt(r_mid**2 + b**2)))

beta = 1.0 - (sig_t**2 / sig_r**2)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8),
                                gridspec_kw={'height_ratios': [3, 1]},
                                sharex=True)

ax1.plot(r_mid, sigma_theory, 'r-', label='Jeans theory')
ax1.scatter(r_mid, sig_r, color='blue',  label='sigma_r  (sim)')
ax1.scatter(r_mid, sig_t, color='green', marker='s', label='sigma_t  (sim, 1D)')
ax1.set_ylabel('velocity dispersion')
ax1.set_title('Jeans equation check — Plummer sphere')
ax1.legend()
ax1.grid(True)

ax2.axhline(0, color='k', ls='--', label='beta=0 (isotropic)')
ax2.scatter(r_mid, beta, color='purple')
ax2.set_ylim(-0.5, 0.5)
ax2.set_xlabel('r [code units]')
ax2.set_ylabel('beta')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("plummer_jeans.pdf")
print("Saved plummer_jeans.pdf")
