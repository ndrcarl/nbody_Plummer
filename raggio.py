import numpy as np
import matplotlib.pyplot as plt
import math

N = 10000
b = 1.0

def lag_radius_theory(frac, b):
    lo, hi = 0.0, 1000.0 * b
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if mid**3 / (mid**2 + b**2)**1.5 < frac:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

r10_th = lag_radius_theory(0.10, b)
r50_th = lag_radius_theory(0.50, b)
r90_th = lag_radius_theory(0.90, b)

def read_radii(filepath):
    times, r10, r50, r90 = [], [], [], []
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
            pos = np.empty((N, 3))
            for j in range(N): pos[j] = np.fromstring(f.readline(), sep=' ')
            for _ in range(N): f.readline()
            for _ in range(N): f.readline()
            cm = np.mean(pos, axis=0)
            radii = np.linalg.norm(pos - cm, axis=1)
            times.append(t)
            r10.append(np.percentile(radii, 10))
            r50.append(np.percentile(radii, 50))
            r90.append(np.percentile(radii, 90))
    return np.array(times), np.array(r10), np.array(r50), np.array(r90)

times, r10, r50, r90 = read_radii("plummer.out")

plt.figure(figsize=(9, 5))
plt.plot(times, r90, 'r-', label='90%% mass  (theory %.3f)' % r90_th)
plt.plot(times, r50, 'b-', label='50%% mass / r_hm  (theory %.3f)' % r50_th)
plt.plot(times, r10, 'g-', label='10%% mass  (theory %.3f)' % r10_th)
plt.axhline(r90_th, color='r', ls='--', alpha=0.4)
plt.axhline(r50_th, color='b', ls='--', alpha=0.4)
plt.axhline(r10_th, color='g', ls='--', alpha=0.4)
plt.xlabel('time [code units]')
plt.ylabel('Lagrangian radius')
plt.title('Lagrangian radii vs time (should be flat)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lagrangian_radii.pdf")
print("Saved lagrangian_radii.pdf")
