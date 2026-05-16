# ============================================================
#  summary_runs.py — Plummer sphere stability analysis
#
#  [PHYSICS OVERVIEW]
#  This script verifies that the N-body simulation perfectly
#  reproduces the analytical physics of a Plummer Sphere (an m=5 Polytrope).
#  It tests macroscopic stability (Virial Theorem, Lagrangian radii),
#  kinematics (Jeans Equation, Isotropy), thermodynamics (Phase-space
#  distribution), and orbital dynamics (Bertrand's Theorem).
#
#  [CODING OVERVIEW]
#  This script uses "Lazy Reading" (streaming). Instead of loading a
#  massive N-body file into RAM, it reads one snapshot at a time,
#  computes the physics metrics, stores the scalar results, and
#  immediately deletes the raw arrays.
#
#  TWO MODES:
#
#  PER-RUN MODE  (called from run_single.sh as):
#    python3 summary_runs.py <run_dir>
#    - chdirs into run_dir
#    - reads plummer.out lazily, one snapshot at a time
#    - saves per-run PDFs inside run_dir
#    - serialises aggregated stats to run_dir/plummer_stats.npz
#
#  COMBINED MODE  (called once after all runs via finalize.sh):
#    python3 summary_runs.py --combined
#    - scans cwd for run_*/plummer_stats.npz
#    - loads only those small npz files (no snapshot re-reading)
#    - produces combined/ plots comparing all realisations
#    - RAM usage: negligible (aggregated arrays only)
#
#  LAZY READING:
#    Snapshot files are streamed one snapshot at a time.
#    Raw arrays (pos, vel, phi) are deleted immediately after use.
#    Only scalar time series are kept in RAM.
#
#  QUANTITIES COMPUTED PER RUN:
#  - Lagrangian radii r_X%(t) at [10,25,50,75,90]%       -- must be flat
#  - Virial ratio 2K/|W|(t)                               -- must be ~1.0
#  - Total velocity dispersion sigma_v(t)                 -- must be flat
#  - Mean radial velocity <v_r>(t)                        -- must be ~0
#  - sigma_t/sigma_r ratio(r) at t=0 and t=final         -- must be ~1.0
#  - Density profile rho(r) at t=0, mid, final           -- must match Plummer
#  - Velocity distribution P(q) at t=0 and t=final       -- must match q^2(1-q^2)^(7/2)
#  - Energy conservation |Delta_E/E|                     -- quality control
#  - 2D snapshots at fixed times (CM-centred)            -- visual check
#  - Orbital rosettes for 3 particles (CM-centred)       -- Bertrand's theorem
#  - Jeans equation check sigma_r, sigma_t vs theory     -- per-run
#  - Cumulative mass M(<r) and circular velocity         -- per-run
#
#  SCALAR STABILITY METRICS (per run, for combined comparison):
#  - Drift of r_hm: (r_hm_final - r_hm_initial) / r_hm_initial
#  - Drift of sigma_v
#  - Mean virial ratio over full run
#  - Mean sigma_t/sigma_r at final snapshot
#  - Max energy conservation error
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import math
import warnings


# ============================================================
#  MODE DETECTION
# ============================================================

COMBINED_MODE = (len(sys.argv) > 1 and sys.argv[1] == "--combined")

if not COMBINED_MODE:
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
    BASE_DIR = os.getcwd()


# ============================================================
#  PHYSICAL PARAMETERS
# ============================================================
# [PHYSICS] N-body units: G=1, M_tot=1. Scale radius b=1.
N     = 10000
M_tot = 1.0
b     = 1.0
G     = 1.0
mass  = M_tot / N

# [PHYSICS] Central density and Dynamical Time (Crossing Time).
rho_c       = 3.0 * M_tot / (4.0 * math.pi * b**3)
t_dyn       = math.sqrt(3.0 * math.pi / (16.0 * G * rho_c))

# [PHYSICS] Half-mass radius for a Plummer sphere is exactly ~1.305 * b.
r_hm_theory = b / math.sqrt(2.0**(2.0/3.0) - 1.0)

LAG_FRACS  = [0.10, 0.25, 0.50, 0.75, 0.90]
LAG_COLORS = ['#2166ac', '#4dac26', '#d7191c', '#fdae61', '#7b3294']

# Times at which to capture 2D snapshots (CM-centred)
SNAP_TARGET_TIMES = [0.0, 50.0, 100.0, 200.0]


# [PHYSICS] Finds the theoretical radii containing specific mass fractions.
# Uses bisection since M(<r)/M_tot = r^3/(r^2+b^2)^{3/2} = frac has no
# closed-form inverse.
def lag_radius_theory(frac, b):
    lo, hi = 0.0, 1000.0 * b
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if mid**3 / (mid**2 + b**2)**1.5 < frac:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

LAG_THEORY = [lag_radius_theory(f, b) for f in LAG_FRACS]

E_ERR_THRESHOLD = 0.01   # [PHYSICS] 1% energy conservation tolerance
HAS_PHI         = True   # True means treecode output the N-body potential


# ============================================================
#  COMBINED MODE
# ============================================================

if COMBINED_MODE:
    BASE_DIR  = os.getcwd()
    npz_files = sorted(glob.glob(os.path.join(BASE_DIR, "run_*", "plummer_stats.npz")))
    if not npz_files:
        print("Combined mode: no run_*/plummer_stats.npz found. Run per-run mode first.")
        sys.exit(1)

    print(f"Combined mode: loading {len(npz_files)} runs.")

    groups = []
    for fpath in npz_files:
        d   = np.load(fpath, allow_pickle=False)
        tag = os.path.basename(os.path.dirname(fpath))
        groups.append({"tag": tag, "d": d})
        print(f"  Loaded: {tag}")

    n_runs  = len(groups)
    t_ref   = groups[0]["d"]["times"]
    b_c     = float(groups[0]["d"]["b"])
    t_dyn_c = float(groups[0]["d"]["t_dyn"])
    r_hm_c  = float(groups[0]["d"]["r_hm_theory"])

    outdir = os.path.join(BASE_DIR, "combined")
    os.makedirs(outdir, exist_ok=True)

    # ---- FIG C1: density profiles overlaid (initial snapshot) ----
    # [PHYSICS] Plummer density: rho(r) = (3M/4pi b^3) * (1 + r^2/b^2)^{-5/2}
    # Flat core for r << b, steep r^{-5} falloff for r >> b.
    max_r_c    = np.max([np.max(g["d"]["rho_r_mid"]) for g in groups])
    r_theory_c = np.logspace(-1.5, np.log10(max_r_c), 300)
    rho_th_c   = (3.0*M_tot/(4.0*math.pi*b_c**3)) * (1.0 + (r_theory_c/b_c)**2)**(-2.5)

    figC1, axC1 = plt.subplots(figsize=(7, 5))
    figC1.suptitle('Density profiles — %d realisations  (initial snapshot)' % n_runs)
    for g in groups:
        axC1.plot(g["d"]["rho_r_mid"], g["d"]["rho_profile"], 'b-', alpha=0.4, lw=1.0)
    axC1.plot(r_theory_c, rho_th_c, 'r-', lw=1.5, label='Plummer theory')
    axC1.axvline(b_c, color='orange', ls='--', label='r = b')
    axC1.set_xscale('log'); axC1.set_yscale('log')
    axC1.set_xlabel('r'); axC1.set_ylabel('density')
    axC1.set_title('blue = individual runs,  red = theory')
    axC1.legend(); axC1.grid(True)
    plt.tight_layout()
    out = os.path.join(outdir, "combined_density.pdf")
    plt.savefig(out); plt.close(figC1)
    print("Saved: %s" % out)

    # ---- FIG C2: circular velocity overlaid (initial snapshot) ----
    # [PHYSICS] v_circ(r) = sqrt(G M(<r) / r) — structural test of IC sampling.
    max_vc_r = np.max([np.max(g["d"]["vcirc_r"]) for g in groups])
    r_vc_th  = np.logspace(-2, np.log10(max_vc_r), 500)
    M_vc_th  = M_tot * r_vc_th**3 / (b_c**3 * (1.0 + (r_vc_th/b_c)**2)**1.5)
    vcirc_th = np.sqrt(G * M_vc_th / r_vc_th)

    figC2, axC2 = plt.subplots(figsize=(7, 5))
    figC2.suptitle('Circular velocity — %d realisations  (initial snapshot)' % n_runs)
    for g in groups:
        axC2.plot(g["d"]["vcirc_r"], g["d"]["vcirc_sim"], 'b-', alpha=0.3, lw=0.8)
    axC2.plot(r_vc_th, vcirc_th, 'r-', lw=1.5, label='Plummer theory')
    axC2.set_xlabel('r'); axC2.set_ylabel('v_circ')
    axC2.set_title('blue = individual runs')
    axC2.legend(); axC2.grid(True)
    plt.tight_layout()
    out = os.path.join(outdir, "combined_vcirc.pdf")
    plt.savefig(out); plt.close(figC2)
    print("Saved: %s" % out)

    # ---- FIG C3: velocity distribution P(q) overlaid (initial snapshot) ----
    # [PHYSICS] P(q) ~ q^2*(1-q^2)^{7/2} — the Eddington DF for the Plummer sphere.
    # Saved from the initial snapshot in the npz to verify IC sampling.
    q_th  = np.linspace(0, 1, 300)
    p_th  = q_th**2 * (1.0 - q_th**2)**3.5
    p_th /= np.trapezoid(p_th, q_th)

    figC3, axC3 = plt.subplots(figsize=(7, 5))
    figC3.suptitle('Velocity distribution P(q) — %d realisations  (initial snapshot)' % n_runs)
    for g in groups:
        axC3.plot(g["d"]["q_mid"], g["d"]["q_hist"], 'b-', alpha=0.4, lw=1.0)
    axC3.plot(q_th, p_th, 'r-', lw=1.5, label='theory q^2(1-q^2)^(7/2)')
    axC3.set_xlabel('q = v / v_esc'); axC3.set_ylabel('probability density')
    axC3.set_title('blue = individual runs')
    axC3.legend(); axC3.grid(True)
    plt.tight_layout()
    out = os.path.join(outdir, "combined_vel_dist.pdf")
    plt.savefig(out); plt.close(figC3)
    print("Saved: %s" % out)

    # ---- FIG C4: Jeans equation and velocity dispersion ratio (initial snapshot) ----
    # [PHYSICS] Isotropic Jeans: sigma_r^2(r) = G M_tot / (6 sqrt(r^2 + b^2))
    # Isotropy check: sigma_t(1D)/sigma_r = 1 for isotropic velocity field.
    # Saved from the initial snapshot in the npz.
    max_j_r = np.max([np.max(g["d"]["jeans_r_mid"]) for g in groups])
    r_j_th  = np.logspace(-1.0, np.log10(max_j_r), 200)
    sig_th  = np.sqrt(G * M_tot / (6.0 * np.sqrt(r_j_th**2 + b_c**2)))

    figC4, (axC4a, axC4b) = plt.subplots(1, 2, figsize=(11, 5))
    figC4.suptitle('Jeans equation and isotropy — %d realisations  (initial snapshot)' % n_runs)
    for g in groups:
        d = g["d"]
        axC4a.scatter(d["jeans_r_mid"], d["jeans_sigr"], s=8, color='blue',  alpha=0.3)
        axC4a.scatter(d["jeans_r_mid"], d["jeans_sigt"], s=8, color='green', alpha=0.3)
        axC4b.plot(d["jeans_r_mid"], d["jeans_ratio"], 'b-', alpha=0.4, lw=1.0)

    axC4a.plot(r_j_th, sig_th, 'r-', lw=1.5, label='Jeans theory')
    axC4a.set_xscale('log')
    axC4a.set_xlabel('r'); axC4a.set_ylabel('velocity dispersion')
    axC4a.set_title('blue = sigma_r,  green = sigma_t')
    axC4a.legend(); axC4a.grid(True)

    axC4b.axhline(1.0, color='k', ls='--', label='σ_t / σ_r = 1 (isotropic)')
    axC4b.axvline(b_c, color='orange', ls=':', label='r=b')
    axC4b.set_xscale('log'); axC4b.set_ylim(0.5, 1.5)
    axC4b.set_xlabel('r'); axC4b.set_ylabel('σ_t(1D) / σ_r')
    axC4b.set_title('Velocity Dispersion Ratio')
    axC4b.legend(); axC4b.grid(True)
    plt.tight_layout()
    out = os.path.join(outdir, "combined_jeans.pdf")
    plt.savefig(out); plt.close(figC4)
    print("Saved: %s" % out)

    # ---- FIG C5: Lagrangian radii ----
    # [PHYSICS] A Lagrangian radius r_X%(t) must be constant for a stable system.
    # Shaded band = min/max envelope across all realisations (Poisson scatter).
    colors_lag = ['blue', 'green', 'red', 'orange', 'purple']
    lag_keys   = [f"lag_{int(f*100):02d}" for f in LAG_FRACS]

    figC5, (axC5a, axC5b) = plt.subplots(1, 2, figsize=(12, 5))
    figC5.suptitle('Lagrangian radii — %d realisations  (b=%.1f, eps=0.012)' % (n_runs, b_c))

    for ki, (key, frac, col, th) in enumerate(zip(lag_keys, LAG_FRACS, colors_lag, LAG_THEORY)):
        mat      = np.array([g["d"][key] for g in groups])
        mat_norm = mat / mat[:, 0:1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med  = np.nanmedian(mat,      axis=0)
            lo   = np.nanmin(mat,         axis=0)
            hi   = np.nanmax(mat,         axis=0)
            medn = np.nanmedian(mat_norm, axis=0)
            lon  = np.nanmin(mat_norm,    axis=0)
            hin  = np.nanmax(mat_norm,    axis=0)
        t_plot = t_ref / t_dyn_c
        label  = 'r_%d%%' % int(frac * 100)
        axC5a.fill_between(t_plot, lo, hi, color=col, alpha=0.15)
        axC5a.plot(t_plot, med, color=col, label=label)
        axC5a.axhline(th, color=col, ls='--', alpha=0.4)
        axC5b.fill_between(t_plot, lon, hin, color=col, alpha=0.15)
        axC5b.plot(t_plot, medn, color=col, label=label)

    axC5a.axhline(r_hm_c, color='k', ls='--', label='r_hm theory = %.3f' % r_hm_c)
    axC5b.axhline(1.0, color='k', ls='--', label='= 1')
    for ax in (axC5a, axC5b):
        ax.set_xlabel('time  [t_dyn]')
        ax.legend(fontsize=8)
        ax.grid(True)
    axC5a.set_ylabel('Lagrangian radius')
    axC5a.set_title('absolute  (band = min/max over runs)')
    axC5b.set_ylabel('r / r(t=0)')
    axC5b.set_title('normalised drift  (= 1 means stable)')
    plt.tight_layout()
    out = os.path.join(outdir, "combined_lagrangian.pdf")
    plt.savefig(out); plt.close(figC5)
    print("Saved: %s" % out)

    # ---- FIG C6: velocity diagnostics ----
    # [PHYSICS] Three stability tests over time:
    # (a) Virial ratio 2K/|W| = 1 (virial theorem)
    # (b) sigma_v flat (total velocity dispersion conserved)
    # (c) <v_r> ~ 0 (no bulk expansion/contraction)
    vr_mat  = np.array([g["d"]["virial_ratio"] for g in groups])
    sv_mat  = np.array([g["d"]["sigma_v"]      for g in groups])
    mvr_mat = np.array([g["d"]["mean_vr"]      for g in groups])

    figC6, (axC6a, axC6b, axC6c) = plt.subplots(1, 3, figsize=(14, 5))
    figC6.suptitle('Velocity diagnostics — %d realisations  (b=%.1f, eps=0.012)' % (n_runs, b_c))

    for ax, mat, ylabel, title, ref in [
        (axC6a, vr_mat,  '2K/|W|', 'virial ratio  (= 1)',     1.0),
        (axC6b, sv_mat,  'sigma_v', 'velocity dispersion',    None),
        (axC6c, mvr_mat, '<v_r>',   'mean radial vel  (= 0)', 0.0),
    ]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(mat, axis=0)
            lo  = np.nanmin(mat,   axis=0)
            hi  = np.nanmax(mat,   axis=0)
        t_plot = t_ref / t_dyn_c
        ax.fill_between(t_plot, lo, hi, color='blue', alpha=0.15)
        ax.plot(t_plot, med, 'b-', label='median over runs')
        if ref is not None:
            ax.axhline(ref, color='k', ls='--', label='= %.1f' % ref)
        ax.set_xlabel('time  [t_dyn]')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True)
    plt.tight_layout()
    out = os.path.join(outdir, "combined_velocities.pdf")
    plt.savefig(out); plt.close(figC6)
    print("Saved: %s" % out)

    # ---- FIG C7: scalar metrics per run ----
    # [PHYSICS] Each run reduced to three scalar stability indicators.
    # Red crosses flag runs that failed energy conservation check.
    run_ids  = np.arange(1, n_runs + 1)
    hm_drift = np.array([float(g["d"]["r_hm_drift"])    for g in groups])
    vir_mean = np.array([float(g["d"]["virial_mean"])   for g in groups])
    sv_drift = np.array([float(g["d"]["sigma_v_drift"]) for g in groups])
    good_arr = np.array([bool(g["d"]["is_good"])        for g in groups])

    figC7, (axC7a, axC7b, axC7c) = plt.subplots(1, 3, figsize=(13, 5))
    figC7.suptitle('Scalar metrics per run — %d realisations  (eps=0.012, b=%.1f)' % (n_runs, b_c))

    for ax, vals, ylabel, title, ref in [
        (axC7a, 100*hm_drift, 'r_hm drift [%]',   'half-mass radius drift  (0% = stable)', 0.0),
        (axC7b, vir_mean,     'mean 2K/|W|',       'time-avg virial ratio  (= 1)',          1.0),
        (axC7c, 100*sv_drift, 'sigma_v drift [%]', 'velocity dispersion drift  (0% = ok)',  0.0),
    ]:
        ax.scatter(run_ids[good_arr],  vals[good_arr],  color='blue', s=50)
        if (~good_arr).any():
            ax.scatter(run_ids[~good_arr], vals[~good_arr], color='red', marker='x', s=80, label='bad run')
        ax.axhline(ref, color='k', ls='--')
        ax.set_xlabel('run #')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True)
    plt.tight_layout()
    out = os.path.join(outdir, "combined_scalars.pdf")
    plt.savefig(out); plt.close(figC7)
    print("Saved: %s" % out)

    # ---- FIG C8: energy distribution overlaid (final snapshot) ----
    # [PHYSICS] E_i = (1/2)v^2 + phi. Bound: E < 0. Unbound (escaped): E > 0.
    # All particles should remain bound for a stable, correctly-sampled sphere.
    figC8, axC8 = plt.subplots(figsize=(7, 5))
    figC8.suptitle('Energy distribution — %d realisations  (final snapshot)' % n_runs)
    for g in groups:
        E = g["d"]["E_final"]
        E_bound   = E[E <= 0]
        E_unbound = E[E > 0]
        if len(E_bound) > 0:
            axC8.hist(E_bound,   bins=50, color='blue', alpha=0.15, density=True)
        if len(E_unbound) > 0:
            axC8.hist(E_unbound, bins=50, color='red',  alpha=0.15, density=True)
    axC8.axvline(0, color='k', ls='--', label='E=0')
    axC8.set_xlabel('specific energy E')
    axC8.set_ylabel('probability density')
    axC8.set_title('blue = bound,  red = unbound')
    axC8.legend(); axC8.grid(True)
    plt.tight_layout()
    out = os.path.join(outdir, "combined_energy.pdf")
    plt.savefig(out); plt.close(figC8)
    print("Saved: %s" % out)

    print(f"\nAll combined plots written to: {outdir}/")
    sys.exit(0)


# ============================================================
#  PER-RUN MODE
# ============================================================

print("=" * 60)
print("PLUMMER SPHERE — STABILITY ANALYSIS")
print(f"  N = {N},  mass per particle = {mass:.6f}")
print(f"  b = {b},  eps = 0.012  (d_mean/10 near r_hm, fixed)")
print(f"  M_tot = {M_tot},  G = {G}")
print(f"  t_dyn = {t_dyn:.4f} code units")
print(f"  r_hm (theory) = {r_hm_theory:.4f} code units")
print(f"  HAS_PHI = {HAS_PHI}")
print("=" * 60)


# ============================================================
#  FILE READER  (lazy generator)
# ============================================================

def iter_snapshots(filepath, N, has_phi=False):
    """
    Generator: yields (t, pos, vel, phi) one snapshot at a time.
    Barnes treecode output format:
        N         <- particle count
        3         <- ndim
        t         <- simulation time (starting time = 0)
        [N lines] <- masses
        [N lines] <- positions (x y z)
        [N lines] <- velocities (vx vy vz)
        [N lines] <- potentials phi  (only if options=out-phi)
    """
    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line: return
            try:
                if int(line.strip()) != N: continue
            except ValueError: continue
            ndim_line = f.readline()
            try:
                if int(ndim_line.strip()) != 3: continue
            except ValueError: continue
            t_line = f.readline()
            if not t_line: return
            try:
                t = float(t_line.strip())
            except ValueError: continue
            for _ in range(N): f.readline()
            pos = np.empty((N, 3))
            for j in range(N): pos[j] = np.fromstring(f.readline(), sep=' ')
            vel = np.empty((N, 3))
            for j in range(N): vel[j] = np.fromstring(f.readline(), sep=' ')
            phi = None
            if has_phi:
                phi = np.empty(N)
                for j in range(N):
                    raw = f.readline()
                    if not raw: return
                    phi[j] = float(raw)
            yield t, pos, vel, phi


# ============================================================
#  HELPER FUNCTIONS
# ============================================================

def lagrangian_radii(pos, cm, fracs):
    """
    Lagrangian radii computed relative to the instantaneous CM.
    For equal-mass particles, counting particles = counting mass.
    """
    r = np.linalg.norm(pos - cm, axis=1)
    r_sorted = np.sort(r)
    result = []
    for f in fracs:
        idx = max(0, min(int(f * len(r_sorted)) - 1, len(r_sorted) - 1))
        result.append(r_sorted[idx])
    return np.array(result)


def velocity_stats(pos, vel):
    """
    Velocity dispersions and mean radial velocity, all in the CM frame.
    Returns: sigma_v (total 3D), sigma_vr (radial), sigma_vt (1D tangential), <v_r>
    """
    pos_cm = np.mean(pos, axis=0)
    vel_cm = np.mean(vel, axis=0)
    pos_c  = pos - pos_cm
    vel_c  = vel - vel_cm
    r_mag  = np.linalg.norm(pos_c, axis=1, keepdims=True)
    r_mag  = np.where(r_mag == 0, 1e-10, r_mag)
    r_hat  = pos_c / r_mag
    vr = np.sum(vel_c * r_hat, axis=1)
    v2 = np.sum(vel_c**2, axis=1)
    vt = np.sqrt(np.maximum(v2 - vr**2, 0.0))
    return np.sqrt(np.mean(v2)), np.std(vr), np.std(vt), np.mean(vr)


def bin_density_profile(pos, cm, mass_per_p, n_bins=35):
    """
    Spherically-averaged density profile in logarithmic radial shells.
    rho = (N_shell * m) / V_shell,  V_shell = (4/3)pi(r2^3 - r1^3)
    Positions are taken relative to the instantaneous CM.
    Returns r_mid, rho, counts (for Poisson error bars).
    """
    r = np.linalg.norm(pos - cm, axis=1)
    if len(r) < 5: return None, None, None
    bins    = np.logspace(np.log10(max(r.min(), 1e-6)), np.log10(r.max()), n_bins)
    counts, edges = np.histogram(r, bins=bins)
    r_mid   = np.sqrt(edges[:-1] * edges[1:])
    V_shell = (4.0/3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    rho     = counts * mass_per_p / V_shell
    ok      = counts > 0
    return r_mid[ok], rho[ok], counts[ok]


def plummer_rho(r, M, b):
    """
    Analytical Plummer density:
      rho(r) = (3M / 4 pi b^3) * (1 + r^2/b^2)^{-5/2}
    """
    return (3.0*M/(4.0*math.pi*b**3)) * (1.0 + (r/b)**2)**(-2.5)


def anisotropy_profile(pos, vel, n_bins=15):
    """
    Compute sigma_t(1D)/sigma_r in radial shells.
    [PHYSICS] For an isotropic velocity distribution this ratio = 1.0 everywhere.
    sigma_t(1D) = sqrt( mean(v_t^2) / 2 ) — the factor 1/2 converts 2D
    tangential variance to one component.
    All velocities computed in the instantaneous CM frame.
    """
    cm     = np.mean(pos, axis=0)
    vel_cm = np.mean(vel, axis=0)
    pos_c  = pos - cm
    vel_c  = vel - vel_cm
    r      = np.linalg.norm(pos_c, axis=1)
    r_hat  = pos_c / np.where(r[:,None] == 0, 1e-10, r[:,None])
    vr     = np.sum(vel_c * r_hat, axis=1)
    vt2    = np.maximum(np.sum(vel_c**2, axis=1) - vr**2, 0.0)
    bins   = np.logspace(np.log10(max(r.min(), 1e-6)), np.log10(r.max()), n_bins + 1)
    r_mid  = np.sqrt(bins[:-1] * bins[1:])
    ratio  = np.full(n_bins, np.nan)
    for k in range(n_bins):
        mask = (r >= bins[k]) & (r < bins[k+1])
        if mask.sum() > 5:
            sig_r = np.std(vr[mask])
            sig_t = np.sqrt(np.mean(vt2[mask]) / 2.0)
            if sig_r > 0:
                ratio[k] = sig_t / sig_r
    ok = np.isfinite(ratio)
    return r_mid[ok], ratio[ok]


def jeans_profile(pos, vel, n_bins=18):
    """
    Compute sigma_r(r), sigma_t(r), and sigma_t/sigma_r(r) in fixed log bins.
    [PHYSICS] Isotropic Jeans prediction:
      sigma_r^2(r) = G M_tot / (6 sqrt(r^2 + b^2))
    Fixed bins [10^{-1}, 10^{1}] ensure comparability across snapshots.
    All velocities computed in the instantaneous CM frame.
    """
    cm     = np.mean(pos, axis=0)
    vel_cm = np.mean(vel, axis=0)
    pos_c  = pos - cm
    vel_c  = vel - vel_cm
    r      = np.linalg.norm(pos_c, axis=1)
    r_hat  = pos_c / np.where(r[:,None] == 0, 1e-10, r[:,None])
    vr     = np.sum(vel_c * r_hat, axis=1)
    vt2    = np.maximum(np.sum(vel_c**2, axis=1) - vr**2, 0.0)
    bins   = np.logspace(-1.0, 1.0, n_bins + 1)
    r_mid  = np.sqrt(bins[:-1] * bins[1:])
    sig_r  = np.full(n_bins, np.nan)
    sig_t  = np.full(n_bins, np.nan)
    ratio  = np.full(n_bins, np.nan)
    for k in range(n_bins):
        mask = (r >= bins[k]) & (r < bins[k+1])
        if mask.sum() > 5:
            sr = np.std(vr[mask])
            st = np.sqrt(np.mean(vt2[mask]) / 2.0)
            sig_r[k] = sr
            sig_t[k] = st
            if sr > 0:
                ratio[k] = st / sr
    return r_mid, sig_r, sig_t, ratio


def get_mid_snapshot(filepath, N, t_target, has_phi=True):
    """
    Second pass over the file to find the snapshot closest to t_target.
    Used only once (for the density profile mid panel).
    """
    best_pos, best_vel, best_phi, best_t = None, None, None, np.inf
    best_diff = np.inf
    for t, pos, vel, phi in iter_snapshots(filepath, N, has_phi=has_phi):
        diff = abs(t - t_target)
        if diff < best_diff:
            best_diff = diff
            best_pos  = pos.copy()
            best_vel  = vel.copy()
            best_phi  = phi.copy() if phi is not None else None
            best_t    = t
    return best_pos, best_vel, best_phi, best_t


# ============================================================
#  MAIN PASS — stream all snapshots
# ============================================================

filepath = "plummer.out"
if not os.path.exists(filepath):
    print(f"Error: {filepath} not found in {os.getcwd()}")
    sys.exit(1)

print(f"\nReading: {filepath}")

times        = []
lag_mat      = []
virial_ratio = []
sigma_v      = []
sigma_vr     = []
sigma_vt     = []
mean_vr_t    = []
energy_abs   = []

pos_initial = vel_initial = phi_initial = None
pos_last    = vel_last    = phi_last    = None

# Orbit selection: track each particle's r_max and r_min over the
# entire run. After the main loop, select 3 particles by their
# orbital turning points rather than their instantaneous position
# at t=0 (which could be at an arbitrary phase of the orbit).
# [PHYSICS] r_max ~ apocentre, r_min ~ pericentre.
# Selection criteria:
#   core particle:      apocentre closest to r_10%  (never leaves the core)
#   half-mass particle: apocentre closest to r_55%  AND
#                       pericentre closest to r_45% (nearly circular at r_hm)
#   halo particle:      pericentre closest to r_90% (never dips inside halo)
# Orbit traces are collected in a dedicated third pass after particle
# selection. CM-subtracted at each step to remove Brownian jitter.
r_max_all    = np.zeros(N)
r_min_all    = np.full(N, np.inf)
r_sum_all    = np.zeros(N)    # for computing time-averaged radius per particle
n_snaps_orb  = 0              # snapshot counter for the mean

# 2D snapshot positions stored CM-centred to prevent visual drift
snap_positions = {}

for t, pos, vel, phi in iter_snapshots(filepath, N, has_phi=HAS_PHI):

    # Instantaneous CM — recomputed every snapshot.
    # Even though sampling_plummer.py zeroed the CM velocity,
    # the tree approximation introduces tiny momentum violations
    # and the statistical CM estimate has noise ~ sigma_r/sqrt(N).
    cm = np.mean(pos, axis=0)

    lr = lagrangian_radii(pos, cm, LAG_FRACS)
    sv, svr, svt, mvr = velocity_stats(pos, vel)

    times.append(t)
    lag_mat.append(lr)
    sigma_v.append(sv)
    sigma_vr.append(svr)
    sigma_vt.append(svt)
    mean_vr_t.append(mvr)

    # [PHYSICS] K = (1/2) sum_i m v_i^2
    # W = (1/2) sum_i m phi_i  (factor 1/2 avoids double-counting)
    # Virial equilibrium: 2K/|W| = 1
    if phi is not None:
        K   = 0.5 * mass * np.sum(np.sum(vel**2, axis=1))
        W   = 0.5 * mass * np.sum(phi)
        vr_ = 2.0 * K / abs(W) if W != 0 else np.nan
        virial_ratio.append(vr_)
        energy_abs.append(abs(K + W))
    else:
        virial_ratio.append(np.nan)
        energy_abs.append(np.nan)

    # Accumulate per-particle r_max, r_min, and r_sum for orbit selection.
    r_cm_all  = np.linalg.norm(pos - cm, axis=1)
    r_max_all = np.maximum(r_max_all, r_cm_all)
    r_min_all = np.minimum(r_min_all, r_cm_all)
    r_sum_all += r_cm_all
    n_snaps_orb += 1

    # [CODING FIX] Save CM-centred positions so 2D snapshots stay
    # visually centred even at late times (t=200).
    for tt in SNAP_TARGET_TIMES:
        if abs(t - tt) < 0.1 and tt not in snap_positions:
            snap_positions[tt] = (pos - cm).copy()

    if pos_initial is None:
        pos_initial = pos.copy()
        vel_initial = vel.copy()
        phi_initial = phi.copy() if phi is not None else None

    pos_last = pos.copy()
    vel_last = vel.copy()
    phi_last = phi.copy() if phi is not None else None

    # Free raw arrays immediately — critical for memory with large files
    del pos, vel, phi

if not times:
    print("Error: no snapshots read.")
    sys.exit(1)

times        = np.array(times)
lag_mat      = np.array(lag_mat)
virial_ratio = np.array(virial_ratio)
sigma_v      = np.array(sigma_v)
sigma_vr     = np.array(sigma_vr)
sigma_vt     = np.array(sigma_vt)
mean_vr_t    = np.array(mean_vr_t)
energy_abs   = np.array(energy_abs)

print(f"Loaded {len(times)} snapshots  "
      f"(t: {times[0]:.4f} -> {times[-1]:.4f},  "
      f"{times[-1]/t_dyn:.1f} t_dyn)")


# ============================================================
#  ORBIT PARTICLE SELECTION — most circular orbit in each region
# ============================================================
# For each particle we know r_max, r_min, and <r> (time-averaged).
# The radial excursion r_max - r_min measures how elliptical the
# orbit is: zero = perfectly circular, large = highly radial.
#
# We define three regions by mass fraction boundaries using <r>
# to classify each particle, then select the one with the smallest
# radial excursion in each region — the most circular orbit there.
#
# Regions:
#   core:      <r> in [0,        r_10%]     (innermost 10% of mass)
#   half-mass: <r> in [r_45%,    r_55%]     (straddles r_hm)
#   halo:      <r> in [r_90%,    inf]       (outermost 10% of mass)
#
# [PHYSICS] A nearly circular orbit in the Plummer potential fills
# a thin annulus in the x-y projection — the cleanest possible
# rosette, making the precession due to Bertrand's theorem most visible.

r_mean_all  = r_sum_all / n_snaps_orb     # time-averaged radius per particle
r_range_all = r_max_all - r_min_all       # radial excursion (0 = circular)

r_45 = lag_radius_theory(0.45, b)
r_55 = lag_radius_theory(0.55, b)

# Region masks based on time-averaged radius
mask_core = r_mean_all <= LAG_THEORY[0]                              # 0 – r_10%
mask_half = (r_mean_all >= r_45) & (r_mean_all <= r_55)             # r_45% – r_55%
mask_halo = r_mean_all >= LAG_THEORY[4]                              # r_90% – inf

# [PHYSICS] Estimate circular orbital period for each particle:
#   T_circ(r) = 2*pi * sqrt( (r^2 + b^2)^{3/2} / (G * M_tot) )
# Particles with T_circ > t_stop / n_min_orbits won't complete enough
# orbits to show a recognisable rosette — exclude them from selection.
# This cap is self-adjusting: it adapts to whatever t_stop was used.
n_min_orbits = 2
T_est        = 2.0 * math.pi * np.sqrt((r_mean_all**2 + b**2)**1.5 / (G * M_tot))
mask_orbit   = T_est < (times[-1] / n_min_orbits)

# Apply orbit-completion constraint to each region
mask_core_ok = mask_core & mask_orbit
mask_half_ok = mask_half & mask_orbit
mask_halo_ok = mask_halo & mask_orbit

# Warn if any region has no particles satisfying both constraints
for name, mask_ok, mask_reg in [
        ('core',      mask_core_ok, mask_core),
        ('half-mass', mask_half_ok, mask_half),
        ('halo',      mask_halo_ok, mask_halo)]:
    if not mask_ok.any():
        print(f"  WARNING: no particle in {name} region completes "
              f"{n_min_orbits} orbits within t_stop={times[-1]:.1f}. "
              f"Falling back to region mask only.")
        if name == 'core':      mask_core_ok = mask_core
        elif name == 'half-mass': mask_half_ok = mask_half
        elif name == 'halo':    mask_halo_ok = mask_halo

# Select minimum-excursion particle in each region
id_core = int(np.argmin(np.where(mask_core_ok, r_range_all, np.inf)))
id_half = int(np.argmin(np.where(mask_half_ok, r_range_all, np.inf)))
id_halo = int(np.argmin(np.where(mask_halo_ok, r_range_all, np.inf)))

tracked_ids  = [id_core, id_half, id_halo]
orbit_colors = ['blue', 'red', 'green']
orbit_labels = [
    'core   <r>={:.3f}  Δr={:.4f}'.format(
        r_mean_all[id_core], r_range_all[id_core]),
    'half-mass  <r>={:.3f}  Δr={:.4f}'.format(
        r_mean_all[id_half], r_range_all[id_half]),
    'halo   <r>={:.3f}  Δr={:.4f}'.format(
        r_mean_all[id_halo], r_range_all[id_halo]),
]

print(f"\nORBIT PARTICLE SELECTION  (minimum radial excursion per region)")
print(f"  {'region':12s}  {'particle':>8s}  {'<r>':>8s}  "
      f"{'r_min':>8s}  {'r_max':>8s}  {'Δr':>8s}  {'N in region':>12s}")
for region, pid, mask in [
        ('core',      id_core, mask_core),
        ('half-mass', id_half, mask_half),
        ('halo',      id_halo, mask_halo)]:
    print(f"  {region:12s}  {pid:8d}  {r_mean_all[pid]:8.4f}  "
          f"{r_min_all[pid]:8.4f}  {r_max_all[pid]:8.4f}  "
          f"{r_range_all[pid]:8.4f}  {mask.sum():12d}")

# Third pass: collect orbit traces for the 3 selected particles.
# [CODING] CM-subtracted at each step for clean rosette plots.
print(f"  Collecting orbit traces (third pass)...")
orbit_x = [[], [], []]
orbit_y = [[], [], []]

for t, pos, vel, phi in iter_snapshots(filepath, N, has_phi=HAS_PHI):
    cm = np.mean(pos, axis=0)
    for i, pid in enumerate(tracked_ids):
        orbit_x[i].append(pos[pid, 0] - cm[0])
        orbit_y[i].append(pos[pid, 1] - cm[1])
    del pos, vel, phi

print(f"  Orbit traces collected.")


# ============================================================
#  QUALITY CONTROL — energy conservation
# ============================================================

energy_err = np.nan
if np.any(np.isfinite(energy_abs)) and energy_abs[0] > 0:
    energy_err = float(np.nanmax(np.abs(energy_abs - energy_abs[0])) / energy_abs[0])
is_good = (np.isnan(energy_err) or energy_err < E_ERR_THRESHOLD)

print(f"\nQUALITY CONTROL")
print(f"  Max |Delta_E/E| = {energy_err:.2e}  "
      f"({'PASS' if is_good else 'FAIL'})")


# ============================================================
#  SCALAR STABILITY METRICS
# ============================================================

r_hm_t       = lag_mat[:, LAG_FRACS.index(0.50)]
r_hm_initial = r_hm_t[0]
r_hm_drift   = (r_hm_t[-1] - r_hm_initial) / r_hm_initial
sv_drift     = (sigma_v[-1] - sigma_v[0]) / sigma_v[0]
virial_mean  = float(np.nanmean(virial_ratio))
virial_std   = float(np.nanstd(virial_ratio))
r_ratio_final, ratio_final = anisotropy_profile(pos_last, vel_last)
ratio_mean = float(np.nanmean(ratio_final)) if len(ratio_final) > 0 else np.nan

print(f"\nSTABILITY SUMMARY")
print(f"  r_hm(t=0)                = {r_hm_initial:.4f}  (theory = {r_hm_theory:.4f})")
print(f"  r_hm drift               = {100*r_hm_drift:+.2f}%")
print(f"  sigma_v drift            = {100*sv_drift:+.2f}%")
print(f"  Mean 2K/|W|              = {virial_mean:.4f} +/- {virial_std:.4f}")
print(f"  Mean σ_t/σ_r (final snap)= {ratio_mean:.4f}  (1.0 is isotropic)")


# ============================================================
#  FIGURE 1: 2D SNAPSHOTS AT FIXED TIMES  (CM-centred)
# ============================================================
# [PHYSICS] Visual check: a stable sphere looks identical at all times.
# [CODING]  Positions stored as (pos - cm) in the main loop, so the
# galaxy stays visually centred even if the CM drifts over t=200.

fig1, axes1 = plt.subplots(1, 4, figsize=(14, 4))
fig1.suptitle('Plummer sphere snapshots (x-y projection, CM-centred, all %d particles)' % N)

for i, tt in enumerate(SNAP_TARGET_TIMES):
    ax = axes1[i]
    if tt not in snap_positions:
        ax.set_title('t = %.1f  (not found)' % tt)
        ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
    else:
        p = snap_positions[tt]
        ax.scatter(p[:, 0], p[:, 1], s=1, color='k', alpha=0.15)
        ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
        ax.set_aspect('equal')
        ax.set_title('t = %.1f' % tt)
    ax.set_xlabel('x')
    if i == 0:
        ax.set_ylabel('y')
    ax.grid(True)

plt.tight_layout()
plt.savefig("summary_snapshots.pdf")
plt.close(fig1)
print("\nSaved: summary_snapshots.pdf")


# ============================================================
#  FIGURE 2: ORBITAL ROSETTES  (CM-centred)
# ============================================================
# [PHYSICS] Bertrand's theorem: only 1/r^2 (Kepler) and r^2 (harmonic)
# potentials produce closed orbits. The Plummer potential is neither,
# so orbits precess → open rosettes filling an annular region.
# [CODING]  Positions are already CM-centred (subtracted in main loop),
# removing the ~0.01 code-unit Brownian jitter that would otherwise
# blur the inner rosette.

fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
fig2.suptitle("Orbital rosettes in Plummer potential — particles selected by turning points (CM-centred)")

for i in range(3):
    axes2[i].plot(orbit_x[i], orbit_y[i], color=orbit_colors[i], lw=0.6)
    axes2[i].plot(0, 0, 'k+', ms=10)
    axes2[i].set_aspect('equal')
    axes2[i].set_title(orbit_labels[i])
    axes2[i].set_xlabel('x')
    axes2[i].grid(True)

axes2[0].set_ylabel('y')
plt.tight_layout()
plt.savefig("summary_orbits.pdf")
plt.close(fig2)
print("Saved: summary_orbits.pdf")


# ============================================================
#  FIGURE 3: DENSITY PROFILE — initial / mid / final
# ============================================================
# [PHYSICS] rho(r) = (3M/4pi b^3)*(1 + r^2/b^2)^{-5/2}
# Three time panels verify the profile is unchanged by the dynamics.
# Error bars = Poisson: sigma_rho = rho / sqrt(N_bin).
# Orange points = r < b (softening-dominated region).

t_mid_target = times[-1] / 2.0
pos_mid, vel_mid, phi_mid, t_mid_actual = get_mid_snapshot(
    filepath, N, t_mid_target, has_phi=HAS_PHI)
print(f"  Density mid snapshot at t = {t_mid_actual:.4f}")

max_r_last = np.max(np.linalg.norm(pos_last - np.mean(pos_last, axis=0), axis=1))
r_theory   = np.logspace(-1.5, np.log10(max_r_last), 300)
rho_th     = plummer_rho(r_theory, M_tot, b)

fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5))
fig3.suptitle('Density profile vs Plummer theory  (b=%.1f, dashed = softening r=b)' % b)

for col_idx, (label, pos_s, t_s) in enumerate([
        ('t=0',                         pos_initial, times[0]),
        ('t=%.2f' % t_mid_actual,        pos_mid,    t_mid_actual),
        ('t=%.2f (final)' % times[-1],  pos_last,    times[-1])]):

    ax   = axes3[col_idx]
    cm_s = np.mean(pos_s, axis=0)
    r_m, rho_m, cnts = bin_density_profile(pos_s, cm_s, mass)

    if r_m is not None:
        inside  = r_m < b
        outside = ~inside
        if outside.any():
            err = rho_m[outside] / np.sqrt(cnts[outside])
            ax.errorbar(r_m[outside], rho_m[outside], yerr=err,
                        fmt='o', ms=3, color='blue', label='N-body')
        if inside.any():
            ax.scatter(r_m[inside], rho_m[inside], s=10, color='orange',
                       label='inside b (softened)')

    ax.plot(r_theory, rho_th, 'r-', label='Plummer theory')
    ax.axvline(b, color='orange', ls='--', label='r = b')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('r')
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.grid(True)

axes3[0].set_ylabel('density')
plt.tight_layout()
plt.savefig("summary_density.pdf")
plt.close(fig3)
print("Saved: summary_density.pdf")


# ============================================================
#  FIGURE 4: CUMULATIVE MASS AND CIRCULAR VELOCITY — initial
# ============================================================
# [PHYSICS] M(<r): sort particle radii, M(<r_i) = i * m.
# v_circ(r) = sqrt(G M(<r) / r) — structural IC check.

_cm_init  = np.mean(pos_initial, axis=0)
_r_init   = np.linalg.norm(pos_initial - _cm_init, axis=1)
_r_sorted = np.sort(_r_init)
_M_cum    = np.arange(1, N + 1) * mass
_mask_vc  = _r_sorted > 0.01
_r_vc     = _r_sorted[_mask_vc]
_vcirc_s  = np.sqrt(G * _M_cum[_mask_vc] / _r_vc)

r_th_vc  = np.logspace(-2, np.log10(np.max(_r_vc)), 500)
M_th_vc  = M_tot * r_th_vc**3 / (b**3 * (1.0 + (r_th_vc/b)**2)**1.5)
vc_th    = np.sqrt(G * M_th_vc / r_th_vc)

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(11, 5))
fig4.suptitle('Cumulative mass and circular velocity — initial snapshot  (t=%.2f)' % times[0])

ax4a.plot(r_th_vc, M_th_vc, 'r-', label='Plummer theory')
ax4a.plot(_r_vc,   _M_cum[_mask_vc], 'b--', lw=0.8, label='N-body')
ax4a.axhline(M_tot, color='k', ls=':', label='M_tot')
ax4a.set_xlabel('r [code units]')
ax4a.set_ylabel('M(<r)')
ax4a.set_title('cumulative mass')
ax4a.legend()
ax4a.grid(True)

ax4b.plot(r_th_vc, vc_th,    'r-',  label='Plummer theory')
ax4b.plot(_r_vc,   _vcirc_s, 'b--', lw=0.8, label='N-body')
ax4b.set_xlabel('r [code units]')
ax4b.set_ylabel('v_circ')
ax4b.set_title('circular velocity')
ax4b.legend()
ax4b.grid(True)

plt.tight_layout()
plt.savefig("summary_mass_vcirc.pdf")
plt.close(fig4)
print("Saved: summary_mass_vcirc.pdf")


# ============================================================
#  FIGURE 5: VELOCITY DISTRIBUTION P(q) — initial and final
# ============================================================
# [PHYSICS] Plummer DF: f(E) ~ (-E)^{7/2} => P(q) ~ q^2*(1-q^2)^{7/2}
# q = v/v_esc(r),  v_esc(r) = sqrt(-2 phi(r))
# t=0 panel verifies IC sampling. t=final panel checks for drift
# due to two-body relaxation or numerical heating.

if phi_initial is not None and phi_last is not None:
    q_theory = np.linspace(0, 1, 300)
    p_theory = q_theory**2 * (1.0 - q_theory**2)**3.5
    p_theory /= np.trapezoid(p_theory, q_theory)

    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(11, 5))
    fig5.suptitle('Velocity distribution P(q) vs theory  (m=5 polytrope DF)')

    for ax, label, vel_s, phi_s in [
            (ax5a, 't=0',                vel_initial, phi_initial),
            (ax5b, 't=%.2f' % times[-1], vel_last,    phi_last)]:

        v_mag = np.linalg.norm(vel_s, axis=1)
        v_esc = np.sqrt(np.maximum(-2.0 * phi_s, 0.0))
        q_sim = np.where(v_esc > 0, v_mag / v_esc, 1.0)
        q_sim = q_sim[q_sim < 1.0]

        ax.hist(q_sim, bins=40, density=True, color='blue', alpha=0.5, label='N-body')
        ax.plot(q_theory, p_theory, 'r-', label='theory q^2(1-q^2)^(7/2)')
        ax.set_xlabel('q = v / v_esc')
        ax.set_ylabel('probability density')
        ax.set_title(label)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("summary_vel_dist.pdf")
    plt.close(fig5)
    print("Saved: summary_vel_dist.pdf")


# ============================================================
#  FIGURE 6: JEANS EQUATION CHECK — initial and final
# ============================================================
# [PHYSICS] Isotropic Jeans equation for Plummer sphere:
#   sigma_r^2(r) = G M_tot / (6 sqrt(r^2 + b^2))
# Lower panel: sigma_t(1D)/sigma_r — must be ~1.0 (isotropy).

sigma_theory_jeans = lambda r_arr: np.sqrt(G * M_tot / (6.0 * np.sqrt(r_arr**2 + b**2)))

fig6, axes6 = plt.subplots(2, 2, figsize=(12, 8),
                            gridspec_kw={'height_ratios': [3, 1]})
fig6.suptitle('Jeans equation check — sigma_r, sigma_t vs theory')

for col, (label, pos_s, vel_s) in enumerate([
        ('t=0',                 pos_initial, vel_initial),
        ('t=%.2f' % times[-1], pos_last,    vel_last)]):

    ax_top = axes6[0, col]
    ax_bot = axes6[1, col]

    r_mid_j, sig_r_j, sig_t_j, ratio_j = jeans_profile(pos_s, vel_s)

    # Theory line spans exactly the data range
    valid = ~np.isnan(sig_r_j)
    if valid.any():
        r_theory_j = np.logspace(np.log10(r_mid_j[valid].min()),
                                  np.log10(r_mid_j[valid].max()), 200)
        sig_th_j   = sigma_theory_jeans(r_theory_j)
        ax_top.plot(r_theory_j, sig_th_j, 'r-', label='Jeans theory')

    ax_top.scatter(r_mid_j, sig_r_j, color='blue',  label='sigma_r (sim)')
    ax_top.scatter(r_mid_j, sig_t_j, color='green', marker='s', label='sigma_t (sim, 1D)')
    ax_top.set_xscale('log')
    ax_top.set_ylabel('velocity dispersion')
    ax_top.set_title(label)
    ax_top.legend(fontsize=8)
    ax_top.grid(True)

    ax_bot.axhline(1.0, color='k', ls='--', label='σ_t(1D) / σ_r = 1 (isotropic)')
    ax_bot.scatter(r_mid_j, ratio_j, color='purple', s=15)
    ax_bot.set_xscale('log')
    ax_bot.set_ylim(0.5, 1.5)
    ax_bot.set_xlabel('r [code units]')
    ax_bot.set_ylabel('σ_t(1D) / σ_r')
    ax_bot.legend(fontsize=8)
    ax_bot.grid(True)

plt.tight_layout()
plt.savefig("summary_jeans.pdf")
plt.close(fig6)
print("Saved: summary_jeans.pdf")


# ============================================================
#  FIGURE 7: LAGRANGIAN RADII vs TIME
# ============================================================
# [PHYSICS] Flat Lagrangian radii = stable equilibrium.
# Secular drift signals: wrong softening, two-body relaxation, or
# ICs out of equilibrium.
# Left: absolute radii vs analytical predictions (dashed).
# Right: normalised to t=0 so drift is immediately visible.

colors_lag = ['blue', 'green', 'red', 'orange', 'purple']

fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(12, 5))
fig7.suptitle('Lagrangian radii vs time  (N=%d, b=%.1f, eps=0.012)' % (N, b))

for k, frac in enumerate(LAG_FRACS):
    col   = colors_lag[k]
    label = 'r_%d%%  (theory %.3f)' % (int(frac*100), LAG_THEORY[k])
    ax7a.plot(times / t_dyn, lag_mat[:, k], color=col, label=label)
    ax7a.axhline(LAG_THEORY[k], color=col, ls='--', alpha=0.4)
    ax7b.plot(times / t_dyn, lag_mat[:, k] / lag_mat[0, k],
              color=col, label='r_%d%%' % int(frac*100))

ax7a.set_xlabel('time  [t_dyn]')
ax7a.set_ylabel('Lagrangian radius')
ax7a.set_title('absolute  (dashed = theory)')
ax7a.legend(fontsize=8)
ax7a.grid(True)

ax7b.axhline(1.0, color='k', ls='--', label='= 1')
ax7b.set_xlabel('time  [t_dyn]')
ax7b.set_ylabel('r / r(t=0)')
ax7b.set_title('normalised drift') # = 1 means stable
ax7b.legend(fontsize=8)
ax7b.grid(True)

plt.tight_layout()
plt.savefig("summary_lagrangian.pdf")
plt.close(fig7)
print("Saved: summary_lagrangian.pdf")


# ============================================================
#  FIGURE 8: VIRIAL RATIO AND VELOCITY DIAGNOSTICS vs TIME
# ============================================================
# [PHYSICS] sigma_v theory from Virial Theorem:
#   W = integral rho*Phi dV = -(3pi/32)*G*M^2/b  (Plummer exact result)
#   2K = |W|  =>  sigma_v = sqrt(3 pi G M / 32 b)
# Note: the naive guess sqrt(GM/2b) is ~30% too large — wrong formula.

sv_theory = math.sqrt(3.0 * math.pi * G * M_tot / (32.0 * b))

fig8, (ax8a, ax8b, ax8c) = plt.subplots(1, 3, figsize=(14, 5))
fig8.suptitle('Velocity diagnostics  (N=%d, b=%.1f,  %.1f t_dyn)' % (N, b, times[-1]/t_dyn))

ax8a.plot(times / t_dyn, virial_ratio, 'b-', label='2K/|W|')
ax8a.axhline(1.0, color='k', ls='--', label='= 1')
ax8a.axhline(virial_mean, color='b', ls=':', label='mean = %.4f' % virial_mean)
ax8a.set_ylim(0.8, 1.2)
ax8a.set_xlabel('time  [t_dyn]')
ax8a.set_ylabel('2K/|W|')
ax8a.set_title('virial ratio')
ax8a.legend()
ax8a.grid(True)

ax8b.plot(times / t_dyn, sigma_v,  'b-',  label='sigma_v total')
ax8b.plot(times / t_dyn, sigma_vr, 'b--', label='sigma_vr')
ax8b.plot(times / t_dyn, sigma_vt, 'b:',  label='sigma_vt (1D)')
ax8b.axhline(sv_theory, color='k', ls='--', label='theory = %.4f' % sv_theory)
ax8b.set_xlabel('time  [t_dyn]')
ax8b.set_ylabel('velocity dispersion')
ax8b.set_title('velocity dispersion')
ax8b.legend()
ax8b.grid(True)

ax8c.plot(times / t_dyn, mean_vr_t, 'g-', label='<v_r>')
ax8c.axhline(0.0, color='k', ls='--', label='= 0')
ax8c.set_xlabel('time  [t_dyn]')
ax8c.set_ylabel('<v_r>')
ax8c.set_title('mean radial velocity  (should be ~0)')
ax8c.legend()
ax8c.grid(True)

plt.tight_layout()
plt.savefig("summary_velocities.pdf")
plt.close(fig8)
print("Saved: summary_velocities.pdf")


# ============================================================
#  FIGURE 9: VELOCITY DISPERSION RATIO — initial and final
# ============================================================
# [PHYSICS] sigma_t(1D)/sigma_r = 1 everywhere for an isotropic sphere.
# Wider adaptive radial range than Fig 6 — useful for detecting
# anisotropy in the halo where Jeans bins are underpopulated.

fig9, (ax9a, ax9b) = plt.subplots(1, 2, figsize=(11, 5))
fig9.suptitle('Velocity Dispersion Ratio') # 1.0 means isotropic

for ax, label, pos_s, vel_s in [
        (ax9a, 't=0',                  pos_initial, vel_initial),
        (ax9b, 't=%.2f' % times[-1],   pos_last,    vel_last)]:

    r_b, ratio_b = anisotropy_profile(pos_s, vel_s)
    if len(r_b) > 0:
        ax.scatter(r_b, ratio_b, color='blue', s=20)
        ax.plot(r_b, ratio_b, 'b-', alpha=0.5)
    ax.axhline(1.0, color='k', ls='--', label='σ_t(1D) / σ_r = 1 (isotropic)')
    ax.axvline(b, color='orange', ls=':', label='r=b (softening)')
    ax.set_xscale('log')
    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel('r')
    ax.set_ylabel('σ_t(1D) / σ_r')
    ax.set_title(label)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("summary_anisotropy.pdf")
plt.close(fig9)
print("Saved: summary_anisotropy.pdf")


# ============================================================
#  FIGURE 10: SCALAR SUMMARY PANEL
# ============================================================
# Quick pass/fail overview:
#   Left:   r_hm(t) — half-mass radius stable?
#   Centre: 2K/|W|(t) — virial theorem satisfied?
#   Right:  |ΔE/E|(t) log scale — integrator accurate?

fig10, (ax10a, ax10b, ax10c) = plt.subplots(1, 3, figsize=(13, 5))
fig10.suptitle('Scalar stability metrics  (N=%d, b=%.1f, eps=0.012,  %.1f t_dyn)' % (
    N, b, times[-1]/t_dyn))

ax10a.plot(times / t_dyn, r_hm_t, 'b-', label='r_hm (sim)')
ax10a.axhline(r_hm_theory, color='r', ls='--', label='theory = %.4f' % r_hm_theory)
ax10a.set_xlabel('time  [t_dyn]')
ax10a.set_ylabel('r_hm')
ax10a.set_title('half-mass radius  (drift = %.2f%%)' % (100*r_hm_drift))
ax10a.legend()
ax10a.grid(True)

ax10b.plot(times / t_dyn, virial_ratio, 'b-', label='2K/|W|')
ax10b.axhline(1.0, color='k', ls='--', label='= 1')
ax10b.set_ylim(0.8, 1.2)
ax10b.set_xlabel('time  [t_dyn]')
ax10b.set_ylabel('2K/|W|')
ax10b.set_title('virial ratio  (mean = %.4f)' % virial_mean)
ax10b.legend()
ax10b.grid(True)

ax10c.set_xlabel('time  [t_dyn]')
if np.any(np.isfinite(energy_abs)) and energy_abs[0] > 0:
    e_err_t = np.abs(energy_abs - energy_abs[0]) / energy_abs[0]
    ax10c.semilogy(times / t_dyn, e_err_t, 'orange', label='|dE/E|')
    ax10c.axhline(E_ERR_THRESHOLD, color='r', ls='--', label='threshold')
    ax10c.legend()
ax10c.set_title('energy conservation  (max = %.2e)' % energy_err)
ax10c.grid(True)

plt.tight_layout()
plt.savefig("summary_scalars.pdf")
plt.close(fig10)
print("Saved: summary_scalars.pdf")


# ============================================================
#  FIGURE 11: ENERGY DISTRIBUTION — bound vs unbound (final)
# ============================================================
# [PHYSICS] E_i = (1/2)|v_i|^2 + phi_i.
# Bound: E < 0.  Unbound (escaped): E > 0.
# A correctly sampled stable sphere keeps essentially all particles bound.
# Significant unbound fraction signals: poor sampling, heating, or
# softening too large.

if phi_last is not None:
    E_final   = 0.5 * np.sum(vel_last**2, axis=1) + phi_last
    n_unbound = int(np.sum(E_final > 0))
    n_bound   = N - n_unbound
    print(f"  Unbound particles: {n_unbound} / {N}  ({100*n_unbound/N:.2f}%)")

    fig11, ax11 = plt.subplots(figsize=(7, 5))
    fig11.suptitle('Energy distribution — final snapshot  (t=%.2f)' % times[-1])

    ax11.hist(E_final[E_final <= 0], bins=50, color='blue', alpha=0.6,
              label='bound  (%d)' % n_bound)
    ax11.hist(E_final[E_final > 0],  bins=50, color='red',  alpha=0.6,
              label='unbound  (%d)' % n_unbound)
    ax11.axvline(0, color='k', ls='--', label='E = 0')
    ax11.set_xlabel('specific energy  E = ½v² + φ')
    ax11.set_ylabel('N particles')
    ax11.set_title('bound vs unbound  (%.2f%% escaped)' % (100*n_unbound/N))
    ax11.legend()
    ax11.grid(True)

    plt.tight_layout()
    plt.savefig("summary_energy.pdf")
    plt.close(fig11)
    print("Saved: summary_energy.pdf")


# ============================================================
#  FINAL SUMMARY TABLE
# ============================================================

print("\n" + "=" * 65)
print("PLUMMER STABILITY — FINAL SUMMARY")
print("=" * 65)
print(f"  Snapshots:           {len(times)}")
print(f"  Time span:           {times[-1]:.4f}  ({times[-1]/t_dyn:.1f} t_dyn)")
print(f"  r_hm(t=0):           {r_hm_initial:.4f}  (theory = {r_hm_theory:.4f})")
print(f"  r_hm drift:          {100*r_hm_drift:+.2f}%")
print(f"  sigma_v(t=0):        {sigma_v[0]:.4f}  (theory = {sv_theory:.4f})")
print(f"  sigma_v drift:       {100*sv_drift:+.2f}%")
print(f"  Mean 2K/|W|:         {virial_mean:.4f} +/- {virial_std:.4f}")
print(f"  Mean σ_t/σ_r (final):{ratio_mean:.4f}  (1.0 is isotropic)")
print(f"  Max |Delta_E/E|:     {energy_err:.2e}  "
      f"({'PASS' if is_good else 'FAIL'})")
if phi_last is not None:
    print(f"  Unbound particles:   {n_unbound} / {N}  ({100*n_unbound/N:.2f}%)")
print("=" * 65)


# ============================================================
#  SERIALISE TO NPZ  (feeds combined mode)
# ============================================================
# All IC-verification profiles (density, Jeans, vdist, vcirc) are
# saved from the INITIAL snapshot so combined mode correctly labels
# them as "initial snapshot" and uses them to compare IC sampling
# quality across realisations.
# Energy distribution is saved from the FINAL snapshot (evaporation check).
# All profiles use FIXED bin edges for consistent cross-run overlay.

lag_keys = {f"lag_{int(f*100):02d}": lag_mat[:, k]
            for k, f in enumerate(LAG_FRACS)}

# --- density profile at INITIAL snapshot (fixed log bins) ---
_cm_i0   = np.mean(pos_initial, axis=0)
_r_i0    = np.linalg.norm(pos_initial - _cm_i0, axis=1)
_bins_d  = np.logspace(np.log10(b * 0.05), np.log10(b * 30), 36)
_cnt, _  = np.histogram(_r_i0, bins=_bins_d)
_vol     = (4.0/3.0) * np.pi * (_bins_d[1:]**3 - _bins_d[:-1]**3)
_rho     = np.where(_cnt > 0, _cnt * mass / _vol, np.nan)
_r_mid_d = np.sqrt(_bins_d[:-1] * _bins_d[1:])

# --- Jeans / ratio profile at INITIAL snapshot (fixed log bins) ---
_r_mid_j, _sigr, _sigt, _ratio_j = jeans_profile(pos_initial, vel_initial, n_bins=20)

# --- circular velocity at INITIAL snapshot ---
_r_sorted2  = np.sort(_r_i0)
_M_cum2     = np.arange(1, N+1) * mass
_mask_vc2   = _r_sorted2 > 0.01
_r_vc2      = _r_sorted2[_mask_vc2]
_vcirc_sim2 = np.sqrt(G * _M_cum2[_mask_vc2] / _r_vc2)

# --- velocity distribution P(q) at INITIAL snapshot (fixed bins) ---
if phi_initial is not None:
    _vmag      = np.linalg.norm(vel_initial, axis=1)
    _vesc      = np.sqrt(np.maximum(-2.0 * phi_initial, 0.0))
    _q_sim     = np.where(_vesc > 0, _vmag / _vesc, 1.0)
    _q_sim     = _q_sim[_q_sim < 1.0]
    _bins_q    = np.linspace(0, 1, 41)
    _q_hist, _ = np.histogram(_q_sim, bins=_bins_q, density=True)
    _q_mid     = 0.5 * (_bins_q[:-1] + _bins_q[1:])
else:
    _q_hist = np.full(40, np.nan)
    _q_mid  = np.linspace(0, 1, 40)

# --- specific energy at FINAL snapshot (evaporation check) ---
if phi_last is not None:
    _E_final = 0.5 * np.sum(vel_last**2, axis=1) + phi_last
else:
    _E_final = np.full(N, np.nan)

np.savez(
    "plummer_stats.npz",
    # physical parameters
    b             = np.array(b),
    t_dyn         = np.array(t_dyn),
    r_hm_theory   = np.array(r_hm_theory),
    # time series
    times         = times,
    virial_ratio  = virial_ratio,
    sigma_v       = sigma_v,
    sigma_vr      = sigma_vr,
    sigma_vt      = sigma_vt,
    mean_vr       = mean_vr_t,
    energy_abs    = energy_abs,
    # scalar stability metrics
    r_hm_drift    = np.array(r_hm_drift),
    sigma_v_drift = np.array(sv_drift),
    virial_mean   = np.array(virial_mean),
    virial_std    = np.array(virial_std),
    ratio_mean    = np.array(ratio_mean),
    energy_err    = np.array(energy_err),
    is_good       = np.array(is_good),
    # IC-verification profiles (initial snapshot, fixed bins)
    rho_r_mid     = _r_mid_d,
    rho_profile   = _rho,
    jeans_r_mid   = _r_mid_j,
    jeans_sigr    = _sigr,
    jeans_sigt    = _sigt,
    jeans_ratio   = _ratio_j,
    vcirc_r       = _r_vc2,
    vcirc_sim     = _vcirc_sim2,
    q_mid         = _q_mid,
    q_hist        = _q_hist,
    # final snapshot
    E_final       = _E_final,
    **lag_keys,
)
print("\nSaved: plummer_stats.npz")
print("Run 'python3 summary_runs.py --combined' from BASE_DIR for cross-run plots.")
