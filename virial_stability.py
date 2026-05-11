import numpy as np
import matplotlib.pyplot as plt

N = 10000
MASS = 0.0001

def read_energies(filepath):
    times, virial_ratio = [], []
    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            try:
                if int(line.strip()) != N: continue
            except ValueError: continue
            f.readline()
            t = float(f.readline().strip())
            for _ in range(N): f.readline()
            for _ in range(N): f.readline()
            vel = np.empty((N, 3))
            for j in range(N): vel[j] = np.fromstring(f.readline(), sep=' ')
            phi = np.empty(N)
            for j in range(N): phi[j] = float(f.readline())
            K = 0.5 * MASS * np.sum(vel**2)
            W = 0.5 * MASS * np.sum(phi)
            times.append(t)
            virial_ratio.append(2.0 * K / abs(W))
    return np.array(times), np.array(virial_ratio)

times, vr = read_energies("plummer.out")

plt.figure(figsize=(9, 5))
plt.plot(times, vr, 'b-', label='2K/|W|')
plt.axhline(1.0, color='k', ls='--', label='virial equilibrium')
plt.ylim(0.8, 1.2)
plt.xlabel('time [code units]')
plt.ylabel('2K/|W|')
plt.title('Virial ratio over time (equilibrium = 1)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("virial_ratio.pdf")
print("Saved virial_ratio.pdf")
