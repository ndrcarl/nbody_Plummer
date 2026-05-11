import numpy as np
import matplotlib.pyplot as plt

N = 10000
MASS = 0.0001

def read_last_snapshot(filepath):
    pos, vel, phi = None, None, None
    with open(filepath, 'r') as f:
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
            phi = np.empty(N)
            for j in range(N): phi[j] = float(f.readline())
    return pos, vel, phi

pos, vel, phi = read_last_snapshot("plummer.out")

# specific energy: E = 1/2 v^2 + phi  (phi already per unit mass)
E = 0.5 * np.sum(vel**2, axis=1) + phi
unbound = E > 0

print("Unbound particles: %d / %d" % (np.sum(unbound), N))

plt.figure(figsize=(7, 5))
plt.hist(E[~unbound], bins=50, color='blue', alpha=0.6, label='bound (%d)' % np.sum(~unbound))
plt.hist(E[unbound],  bins=50, color='red',  alpha=0.6, label='unbound (%d)' % np.sum(unbound))
plt.axvline(0, color='k', ls='--')
plt.xlabel('specific energy E')
plt.ylabel('N particles')
plt.title('Energy distribution at final snapshot')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("energy_distribution.pdf")
print("Saved energy_distribution.pdf")
