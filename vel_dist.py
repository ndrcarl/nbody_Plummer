import numpy as np
import matplotlib.pyplot as plt
import sys

# ============================================================
#  vel_dist.py
#  Verifies the phase-space distribution matches P(q) ∝ q^2(1-q^2)^3.5
# ============================================================

N = 10000

# Use the file passed in the terminal, OR default to run_001
filename = sys.argv[1] if len(sys.argv) > 1 else "run_001/plummer.out"

# Read the last snapshot
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

# Calculate dimensionless velocity q
v_mag = np.linalg.norm(vel, axis=1)
v_esc = np.sqrt(-2.0 * phi)  # phi is negative, so -2*phi is positive
q = v_mag / v_esc

# Filter out any numerical anomalies (q should be strictly < 1)
q = q[q < 1.0]

# Analytical curve from your notes
q_theory = np.linspace(0, 1, 200)
p_theory = (q_theory**2) * (1.0 - q_theory**2)**3.5

# Normalize the analytical curve numerically to match histogram area
# FIXED: np.trapz was renamed to np.trapezoid in NumPy 2.0+
p_theory /= np.trapezoid(p_theory, q_theory)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(q, bins=40, density=True, color='steelblue', alpha=0.7, edgecolor='black', label='N-body Simulation')

# FIXED: Added 'r' before the string to make it a raw string for LaTeX
ax.plot(q_theory, p_theory, color='crimson', lw=2.5, label=r'$P(q) \propto q^2(1-q^2)^{7/2}$')

ax.set_xlabel('Dimensionless Velocity $q = v / v_{esc}$')
ax.set_ylabel('Probability Density')
ax.set_title('Phase-Space Velocity Distribution (Rejection Method Proof)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plummer_velocity_dist.pdf", dpi=200)
print(f"Saved plummer_velocity_dist.pdf using data from {filename}")
