import numpy as np
import random
import math

# ============================================================
#  sampling_plummer.py
#  Generates equilibrium initial conditions for a Plummer sphere
#  using Inverse Transform (positions) and Rejection Sampling (velocities).
# ============================================================

N = 10000
M_tot = 1.0
b = 1.0        # Plummer scale radius = system size (code units)
G = 1.0
mass_i = M_tot / N

# Derived quantities (for reference only — eps is set in run.sh)
R_hm   = b / math.sqrt(2**(2/3) - 1.0)   # ~1.305 b
V_hm   = (4.0 / 3.0) * math.pi * R_hm**3
n_hm   = (N / 2.0) / V_hm
d_mean = n_hm**(-1.0/3.0)

print(f"--- Plummer Sphere Setup ---")
print(f"b (scale radius): {b}")
print(f"Half-mass radius: {R_hm:.4f}")
print(f"Mean separation near r_hm: {d_mean:.4f}  ->  eps ~ {d_mean/10:.4f} (set in run.sh)")

def isotropic_vec(mag):
    """Generate a random 3D vector with a given magnitude."""
    u = random.random()
    w = random.random()
    theta = math.acos(1.0 - 2.0 * u)
    phi = 2.0 * math.pi * w
    return (mag * math.sin(theta) * math.cos(phi),
            mag * math.sin(theta) * math.sin(phi),
            mag * math.cos(theta))

def get_q():
    """Rejection sampling for the dimensionless velocity q = v/v_esc"""
    # True maximum of g(q) = q^2*(1-q^2)^(7/2) is at q* = sqrt(2/9)
    # g(q*) = (2/9)*(7/9)^(7/2) = 0.09227...  Use 0.093 to guarantee
    # the envelope always lies above g(q) everywhere in [0,1].
    g_max = 0.093
    while True:
        q = random.random()
        g = (q**2) * ((1.0 - q**2)**3.5)
        if random.random() * g_max < g:
            return q

positions = []
velocities =[]

for _ in range(N):
    # 1. Sample Position (Inverse Transform)
    X = random.random()
    r = b / math.sqrt(X**(-2.0/3.0) - 1.0)
    x, y, z = isotropic_vec(r)
    positions.append([x, y, z])
    
    # 2. Sample Velocity (Rejection Method)
    Psi = G * M_tot / math.sqrt(r**2 + b**2)
    v_esc = math.sqrt(2.0 * Psi)
    q = get_q()
    v = q * v_esc
    vx, vy, vz = isotropic_vec(v)
    velocities.append([vx, vy, vz])

positions = np.array(positions)
velocities = np.array(velocities)

# Shift to exact Center of Mass frame to prevent bulk drift
pos_cm = np.mean(positions, axis=0)
vel_cm = np.mean(velocities, axis=0)
positions -= pos_cm
velocities -= vel_cm

# Write to treecode format
with open("plummer.txt", "w") as f:
    f.write(f"{N}\n3\n0\n")
    for _ in range(N):
        f.write(f"{mass_i}\n")
    for p in positions:
        f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
    for v in velocities:
        f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")

print("Saved initial conditions to plummer.txt")
