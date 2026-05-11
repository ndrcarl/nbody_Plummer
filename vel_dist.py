import numpy as np
import matplotlib.pyplot as plt
import sys

N = 10000

filename = sys.argv[1] if len(sys.argv) > 1 else "plummer.out"

vel, phi = None, None
with open(filename, 'r') as f:
    while True:
        line = f.readline()
        if not line: break
        try:
            if int(line.strip()) != N: continue
        except ValueError: continue
        f.readline(); f.readline()
        for _ in range(N): f.readline()
        for _ in range(N): f.readline()
        vel = np.empty((N, 3))
        for j in range(N): vel[j] = np.fromstring(f.readline(), sep=' ')
        phi = np.empty(N)
        for j in range(N): phi[j] = float(f.readline())

# q = v / v_esc,  v_esc = sqrt(-2*phi)  (phi is negative)
v_mag = np.linalg.norm(vel, axis=1)
v_esc = np.sqrt(-2.0 * phi)
q = v_mag / v_esc
q = q[q < 1.0]   # keep only bound particles

q_theory = np.linspace(0, 1, 200)
p_theory = q_theory**2 * (1.0 - q_theory**2)**3.5
p_theory /= np.trapezoid(p_theory, q_theory)

plt.figure(figsize=(7, 5))
plt.hist(q, bins=40, density=True, color='blue', alpha=0.5, label='N-body')
plt.plot(q_theory, p_theory, 'r-', label=r'$q^2 (1-q^2)^{7/2}$ (theory)')
plt.xlabel('q = v / v_esc')
plt.ylabel('probability density')
plt.title('Velocity distribution (rejection sampling check)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plummer_velocity_dist.pdf")
print("Saved plummer_velocity_dist.pdf")
