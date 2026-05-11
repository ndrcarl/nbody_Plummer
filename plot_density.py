import numpy as np
import matplotlib.pyplot as plt
import sys

N = 10000
M_tot = 1.0
b = 1.0

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
r = np.linalg.norm(pos - cm, axis=1)

bins = np.logspace(-1.5, 1.5, 30)
counts, edges = np.histogram(r, bins=bins)
volumes = (4.0/3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)
rho_sim = (counts * (M_tot / N)) / volumes
r_mid = np.sqrt(edges[1:] * edges[:-1])

rho_theory = (3.0 * M_tot / (4.0 * np.pi * b**3)) * (1.0 + (r_mid/b)**2)**(-2.5)

plt.figure(figsize=(7, 5))
plt.scatter(r_mid, rho_sim, s=20, color='blue', label='N-body')
plt.plot(r_mid, rho_theory, 'r-', label='Plummer theory')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('r [code units]')
plt.ylabel('density')
plt.title('Density profile vs Plummer (m=5 polytrope)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plummer_density.pdf")
print("Saved plummer_density.pdf")
