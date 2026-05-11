import numpy as np
import matplotlib.pyplot as plt
import sys

N = 10000
M_tot = 1.0
b = 1.0
G = 1.0
mass_per_particle = M_tot / N

filename = sys.argv[1] if len(sys.argv) > 1 else "plummer.out"

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

cm = np.mean(pos, axis=0)
r  = np.linalg.norm(pos - cm, axis=1)

r_sorted = np.sort(r)
M_sim    = np.arange(1, N + 1) * mass_per_particle

mask     = r_sorted > 0.01
r_clean  = r_sorted[mask]
M_clean  = M_sim[mask]
v_circ_sim = np.sqrt(G * M_clean / r_clean)

r_theory      = np.logspace(-2, 1.5, 500)
M_theory      = M_tot * r_theory**3 / (b**3 * (1.0 + (r_theory/b)**2)**1.5)
v_circ_theory = np.sqrt(G * M_theory / r_theory)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle('Plummer sphere: cumulative mass and circular velocity')

ax1.plot(r_theory, M_theory, 'r-', label='theory')
ax1.plot(r_clean,  M_clean,  'b--', label='N-body')
ax1.axhline(M_tot, color='k', ls=':', label='M_tot')
ax1.set_xlabel('r [code units]')
ax1.set_ylabel('M(<r)')
ax1.set_title('cumulative mass')
ax1.legend()
ax1.grid(True)

ax2.plot(r_theory, v_circ_theory, 'r-', label='theory')
ax2.plot(r_clean,  v_circ_sim,    'b--', label='N-body')
ax2.set_xlabel('r [code units]')
ax2.set_ylabel('v_circ')
ax2.set_title('circular velocity')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("plummer_mass_vcirc.pdf")
print("Saved plummer_mass_vcirc.pdf")
