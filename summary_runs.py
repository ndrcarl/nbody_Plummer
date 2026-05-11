# ============================================================
#  summary_runs.py — Plummer sphere stability analysis
#
#  TWO MODES:
#
#  PER-RUN MODE  (called from run.sh as):
#    python3 summary_runs.py <run_dir>
#    - chdirs into run_dir
#    - reads plummer.out lazily, one snapshot at a time
#    - saves per-run PDFs inside run_dir
#    - serialises aggregated stats to run_dir/plummer_stats.npz
#
#  COMBINED MODE  (called once after all runs):
#    python3 summary_runs.py --combined
#    - scans cwd for run_*/plummer_stats.npz
#    - loads only those small npz files (no snapshot re-reading)
#    - produces combined/ plots comparing all realisations at the
#      same fixed eps=0.012, quantifying run-to-run scatter from
#      Poisson noise in IC sampling (not an eps sweep)
#    - RAM usage: negligible (aggregated arrays only)
#
#  LAZY READING:
#    Snapshot files are streamed one snapshot at a time.
#    Raw arrays (pos, vel, phi) are deleted immediately after use.
#    Only scalar time series are kept in RAM.
#
#  QUANTITIES COMPUTED PER RUN:
#  - Lagrangian radii r_X%(t) at [10,25,50,75,90]% -- must be flat
#  - Virial ratio 2K/|W|(t)                         -- must be ~1.0 always
#  - Total velocity dispersion sigma_v(t)            -- must be flat
#  - Mean radial velocity <v_r>(t)                   -- must be ~0
#  - Anisotropy beta(r) at t=0 and t=final           -- must be ~0
#  - Density profile rho(r) at t=0, mid, final       -- must match Plummer
#  - Velocity distribution P(q) at t=0 and t=final  -- must match f~eps^(7/2)
#  - Energy conservation |Delta_E/E|                 -- quality control
#
#  SCALAR STABILITY METRICS (per run, for combined comparison):
#  - Drift of r_hm: (r_hm_final - r_hm_initial) / r_hm_initial
#  - Drift of sigma_v
#  - Mean virial ratio over full run
#  - Mean anisotropy beta at final snapshot
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

N     = 10000
M_tot = 1.0
b     = 1.0        # Plummer scale radius = system size
G     = 1.0
mass  = M_tot / N

rho_c       = 3.0 * M_tot / (4.0 * math.pi * b**3)
t_dyn       = math.sqrt(3.0 * math.pi / (16.0 * G * rho_c))
r_hm_theory = b / math.sqrt(2.0**(2.0/3.0) - 1.0)   # ~1.305

LAG_FRACS  = [0.10, 0.25, 0.50, 0.75, 0.90]
LAG_COLORS = ['#2166ac', '#4dac26', '#d7191c', '#fdae61', '#7b3294']

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

E_ERR_THRESHOLD = 0.01
HAS_PHI         = True


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

    # ---- FIG C1: Lagrangian radii ----
    colors_lag = ['blue', 'green', 'red', 'orange', 'purple']
    lag_keys   = [f"lag_{int(f*100):02d}" for f in LAG_FRACS]

    figC1, (axC1a, axC1b) = plt.subplots(1, 2, figsize=(12, 5))
    figC1.suptitle('Lagrangian radii — %d realisations  (b=%.1f, eps=0.012)' % (n_runs, b_c))

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
        axC1a.fill_between(t_plot, lo, hi, color=col, alpha=0.15)
        axC1a.plot(t_plot, med, color=col, label=label)
        axC1a.axhline(th, color=col, ls='--', alpha=0.4)
        axC1b.fill_between(t_plot, lon, hin, color=col, alpha=0.15)
        axC1b.plot(t_plot, medn, color=col, label=label)

    axC1a.axhline(r_hm_c, color='k', ls='--', label='r_hm theory = %.3f' % r_hm_c)
    axC1b.axhline(1.0, color='k', ls='--', label='= 1')
    for ax in (axC1a, axC1b):
        ax.set_xlabel('time  [t_dyn]')
        ax.legend(fontsize=8)
        ax.grid(True)
    axC1a.set_ylabel('Lagrangian radius')
    axC1a.set_title('absolute  (band = min/max over runs)')
    axC1b.set_ylabel('r / r(t=0)')
    axC1b.set_title('normalised drift  (= 1 means stable)')

    plt.tight_layout()
    out = os.path.join(outdir, "combined_lagrangian.pdf")
    plt.savefig(out); plt.close(figC1)
    print("Saved: %s" % out)

    # ---- FIG C2: velocity diagnostics ----
    vr_mat  = np.array([g["d"]["virial_ratio"] for g in groups])
    sv_mat  = np.array([g["d"]["sigma_v"]      for g in groups])
    mvr_mat = np.array([g["d"]["mean_vr"]      for g in groups])

    figC2, (axC2a, axC2b, axC2c) = plt.subplots(1, 3, figsize=(14, 5))
    figC2.suptitle('Velocity diagnostics — %d realisations  (b=%.1f, eps=0.012)' % (n_runs, b_c))

    for ax, mat, ylabel, title, ref in [
        (axC2a, vr_mat,  '2K/|W|',    'virial ratio  (= 1)',        1.0),
        (axC2b, sv_mat,  'sigma_v',   'velocity dispersion',        None),
        (axC2c, mvr_mat, '<v_r>',     'mean radial vel  (= 0)',     0.0),
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
    plt.savefig(out); plt.close(figC2)
    print("Saved: %s" % out)

    # ---- FIG C3: scalar metrics per run ----
    run_ids  = np.arange(1, n_runs + 1)
    hm_drift = np.array([float(g["d"]["r_hm_drift"])    for g in groups])
    vir_mean = np.array([float(g["d"]["virial_mean"])   for g in groups])
    sv_drift = np.array([float(g["d"]["sigma_v_drift"]) for g in groups])
    good_arr = np.array([bool(g["d"]["is_good"])        for g in groups])

    figC3, (axC3a, axC3b, axC3c) = plt.subplots(1, 3, figsize=(13, 5))
    figC3.suptitle('Scalar metrics per run — %d realisations  (eps=0.012, b=%.1f)' % (n_runs, b_c))

    for ax, vals, ylabel, title, ref in [
        (axC3a, 100*hm_drift, 'r_hm drift [%]',  'half-mass radius drift  (0% = stable)', 0.0),
        (axC3b, vir_mean,     'mean 2K/|W|',      'time-avg virial ratio  (= 1)',          1.0),
        (axC3c, 100*sv_drift, 'sigma_v drift [%]','velocity dispersion drift  (0% = ok)',  0.0),
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
    plt.savefig(out); plt.close(figC3)
    print("Saved: %s" % out)

    print(f"\nAll combined plots written to: {outdir}/")

    # ---- FIG C4: density profiles overlaid ----
    r_theory_c  = np.logspace(np.log10(b_c * 0.05), np.log10(b_c * 30), 300)
    rho_th_c    = (3.0*M_tot/(4.0*math.pi*b_c**3)) * (1.0 + (r_theory_c/b_c)**2)**(-2.5)

    figC4, ax4 = plt.subplots(figsize=(7, 5))
    figC4.suptitle('Density profiles — %d realisations  (final snapshot)' % n_runs)
    for g in groups:
        ax4.plot(g["d"]["rho_r_mid"], g["d"]["rho_profile"],
                 'b-', alpha=0.4, lw=1.0)
    ax4.plot(r_theory_c, rho_th_c, 'r-', lw=1.5, label='Plummer theory')
    ax4.axvline(b_c, color='orange', ls='--', label='r = b')
    ax4.set_xscale('log'); ax4.set_yscale('log')
    ax4.set_xlabel('r'); ax4.set_ylabel('density')
    ax4.set_title('blue = individual runs,  red = theory')
    ax4.legend(); ax4.grid(True)
    plt.tight_layout()
    out = os.path.join(outdir, "combined_density.pdf")
    plt.savefig(out); plt.close(figC4)
    print("Saved: %s" % out)

    # ---- FIG C5: Jeans and beta overlaid ----
    r_j_th  = np.logspace(-1.0, 1.0, 100)
    sig_th  = np.sqrt(G * M_tot / (6.0 * np.sqrt(r_j_th**2 + b_c**2)))

    figC5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(11, 5))
    figC5.suptitle('Jeans equation and anisotropy — %d realisations  (final snapshot)' % n_runs)
    for g in groups:
        d = g["d"]
        ax5a.scatter(d["jeans_r_mid"], d["jeans_sigr"], s=8, color='blue',  alpha=0.3)
        ax5a.scatter(d["jeans_r_mid"], d["jeans_sigt"], s=8, color='green', alpha=0.3)
        ax5b.plot(d["jeans_r_mid"], d["jeans_beta"], 'b-', alpha=0.4, lw=1.0)
    ax5a.plot(r_j_th, sig_th, 'r-', lw=1.5, label='Jeans theory')
    ax5a.set_xscale('log')
    ax5a.set_xlabel('r'); ax5a.set_ylabel('velocity dispersion')
    ax5a.set_title('blue = sigma_r,  green = sigma_t')
    ax5a.legend(); ax5a.grid(True)
    ax5b.axhline(0.0, color='k', ls='--', label='beta=0 (isotropic)')
    ax5b.axvline(b_c, color='orange', ls=':', label='r=b')
    ax5b.set_xscale('log'); ax5b.set_ylim(-1.0, 1.0)
    ax5b.set_xlabel('r'); ax5b.set_ylabel('beta')
    ax5b.set_title('anisotropy (should be ~0)')
    ax5b.legend(); ax5b.grid(True)
    plt.tight_layout()
    out = os.path.join(outdir, "combined_jeans.pdf")
    plt.savefig(out); plt.close(figC5)
    print("Saved: %s" % out)

    # ---- FIG C6: circular velocity overlaid ----
    r_vc_th     = np.logspace(-2, 1.5, 500)
    M_vc_th     = M_tot * r_vc_th**3 / (b_c**3 * (1.0 + (r_vc_th/b_c)**2)**1.5)
    vcirc_th    = np.sqrt(G * M_vc_th / r_vc_th)

    figC6, ax6 = plt.subplots(figsize=(7, 5))
    figC6.suptitle('Circular velocity — %d realisations  (final snapshot)' % n_runs)
    for g in groups:
        ax6.plot(g["d"]["vcirc_r"], g["d"]["vcirc_sim"], 'b-', alpha=0.3, lw=0.8)
    ax6.plot(r_vc_th, vcirc_th, 'r-', lw=1.5, label='Plummer theory')
    ax6.set_xlabel('r'); ax6.set_ylabel('v_circ')
    ax6.set_title('blue = individual runs')
    ax6.legend(); ax6.grid(True)
    plt.tight_layout()
    out = os.path.join(outdir, "combined_vcirc.pdf")
    plt.savefig(out); plt.close(figC6)
    print("Saved: %s" % out)

    # ---- FIG C7: velocity distribution P(q) overlaid ----
    q_th     = np.linspace(0, 1, 300)
    p_th     = q_th**2 * (1.0 - q_th**2)**3.5
    p_th    /= np.trapezoid(p_th, q_th)

    figC7, ax7 = plt.subplots(figsize=(7, 5))
    figC7.suptitle('Velocity distribution P(q) — %d realisations  (final snapshot)' % n_runs)
    for g in groups:
        ax7.plot(g["d"]["q_mid"], g["d"]["q_hist"], 'b-', alpha=0.4, lw=1.0)
    ax7.plot(q_th, p_th, 'r-', lw=1.5, label='theory q^2(1-q^2)^(7/2)')
    ax7.set_xlabel('q = v / v_esc'); ax7.set_ylabel('probability density')
    ax7.set_title('blue = individual runs')
    ax7.legend(); ax7.grid(True)
    plt.tight_layout()
    out = os.path.join(outdir, "combined_vel_dist.pdf")
    plt.savefig(out); plt.close(figC7)
    print("Saved: %s" % out)

    # ---- FIG C8: energy distribution overlaid ----
    figC8, ax8 = plt.subplots(figsize=(7, 5))
    figC8.suptitle('Energy distribution — %d realisations  (final snapshot)' % n_runs)
    for g in groups:
        E = g["d"]["E_final"]
        E_bound   = E[E <= 0]
        E_unbound = E[E > 0]
        if len(E_bound) > 0:
            ax8.hist(E_bound,   bins=50, color='blue', alpha=0.15, density=True)
        if len(E_unbound) > 0:
            ax8.hist(E_unbound, bins=50, color='red',  alpha=0.15, density=True)
    ax8.axvline(0, color='k', ls='--', label='E=0')
    ax8.set_xlabel('specific energy E')
    ax8.set_ylabel('probability density')
    ax8.set_title('blue = bound,  red = unbound')
    ax8.legend(); ax8.grid(True)
    plt.tight_layout()
    out = os.path.join(outdir, "combined_energy.pdf")
    plt.savefig(out); plt.close(figC8)
    print("Saved: %s" % out)

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
#  FILE READER
# ============================================================

def iter_snapshots(filepath, N, has_phi=False):
    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                return
            try:
                if int(line.strip()) != N:
                    continue
            except ValueError:
                continue
            ndim_line = f.readline()
            try:
                if int(ndim_line.strip()) != 3:
                    continue
            except ValueError:
                continue
            t_line = f.readline()
            if not t_line:
                return
            try:
                t = float(t_line.strip())
            except ValueError:
                continue
            for _ in range(N): f.readline()
            pos = np.empty((N, 3))
            for j in range(N):
                pos[j] = np.fromstring(f.readline(), sep=' ')
            vel = np.empty((N, 3))
            for j in range(N):
                vel[j] = np.fromstring(f.readline(), sep=' ')
            phi = None
            if has_phi:
                phi = np.empty(N)
                for j in range(N):
                    raw = f.readline()
                    if not raw:
                        return
                    phi[j] = float(raw)
            yield t, pos, vel, phi


# ============================================================
#  HELPERS
# ============================================================

def lagrangian_radii(pos, cm, fracs):
    r = np.linalg.norm(pos - cm, axis=1)
    r_sorted = np.sort(r)
    result = []
    for f in fracs:
        idx = max(0, min(int(f * len(r_sorted)) - 1, len(r_sorted) - 1))
        result.append(r_sorted[idx])
    return np.array(result)


def velocity_stats(pos, vel):
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
    r = np.linalg.norm(pos - cm, axis=1)
    if len(r) < 5:
        return None, None, None
    bins = np.logspace(np.log10(max(r.min(), 1e-6)),
                       np.log10(r.max()), n_bins)
    counts, edges = np.histogram(r, bins=bins)
    r_mid   = np.sqrt(edges[:-1] * edges[1:])
    V_shell = (4.0/3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    rho     = counts * mass_per_p / V_shell
    ok      = counts > 0
    return r_mid[ok], rho[ok], counts[ok]


def plummer_rho(r, M, b):
    return (3.0*M/(4.0*math.pi*b**3)) * (1.0 + (r/b)**2)**(-2.5)


def anisotropy_profile(pos, vel, n_bins=15):
    cm     = np.mean(pos, axis=0)
    vel_cm = np.mean(vel, axis=0)
    pos_c  = pos - cm
    vel_c  = vel - vel_cm
    r      = np.linalg.norm(pos_c, axis=1)
    r_hat  = pos_c / np.where(r[:,None] == 0, 1e-10, r[:,None])
    vr     = np.sum(vel_c * r_hat, axis=1)
    vt2    = np.maximum(np.sum(vel_c**2, axis=1) - vr**2, 0.0)
    bins   = np.logspace(np.log10(max(r.min(), 1e-6)),
                         np.log10(r.max()), n_bins + 1)
    r_mid  = np.sqrt(bins[:-1] * bins[1:])
    beta   = np.full(n_bins, np.nan)
    for k in range(n_bins):
        mask = (r >= bins[k]) & (r < bins[k+1])
        if mask.sum() > 5:
            sig_r = np.std(vr[mask])
            sig_t = np.sqrt(np.mean(vt2[mask]) / 2.0)
            if sig_r > 0:
                beta[k] = 1.0 - (sig_t**2 / sig_r**2)
    ok = np.isfinite(beta)
    return r_mid[ok], beta[ok]


def get_mid_snapshot(filepath, N, t_target, has_phi=True):
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

for t, pos, vel, phi in iter_snapshots(filepath, N, has_phi=HAS_PHI):
    cm = np.mean(pos, axis=0)
    lr = lagrangian_radii(pos, cm, LAG_FRACS)
    sv, svr, svt, mvr = velocity_stats(pos, vel)

    times.append(t)
    lag_mat.append(lr)
    sigma_v.append(sv)
    sigma_vr.append(svr)
    sigma_vt.append(svt)
    mean_vr_t.append(mvr)

    if phi is not None:
        K   = 0.5 * mass * np.sum(np.sum(vel**2, axis=1))
        W   = 0.5 * mass * np.sum(phi)
        vr_ = 2.0 * K / abs(W) if W != 0 else np.nan
        virial_ratio.append(vr_)
        energy_abs.append(abs(K + W))
    else:
        virial_ratio.append(np.nan)
        energy_abs.append(np.nan)

    if pos_initial is None:
        pos_initial = pos.copy()
        vel_initial = vel.copy()
        phi_initial = phi.copy() if phi is not None else None

    pos_last = pos.copy()
    vel_last = vel.copy()
    phi_last = phi.copy() if phi is not None else None
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
#  QUALITY CONTROL
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
r_beta_final, beta_final = anisotropy_profile(pos_last, vel_last)
beta_mean = float(np.nanmean(beta_final)) if len(beta_final) > 0 else np.nan

print(f"\nSTABILITY SUMMARY")
print(f"  r_hm(t=0)                = {r_hm_initial:.4f}  (theory = {r_hm_theory:.4f})")
print(f"  r_hm drift               = {100*r_hm_drift:+.2f}%")
print(f"  sigma_v drift            = {100*sv_drift:+.2f}%")
print(f"  Mean 2K/|W|              = {virial_mean:.4f} +/- {virial_std:.4f}")
print(f"  Mean beta (final snap)   = {beta_mean:.4f}  (0 = isotropic)")


# ============================================================
#  FIGURE 1: LAGRANGIAN RADII vs TIME
# ============================================================

colors_lag = ['blue', 'green', 'red', 'orange', 'purple']

fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle('Lagrangian radii vs time  (N=%d, b=%.1f, eps=0.012)' % (N, b))

for k, frac in enumerate(LAG_FRACS):
    col = colors_lag[k]
    label = 'r_%d%%  (theory %.3f)' % (int(frac*100), LAG_THEORY[k])
    ax1a.plot(times / t_dyn, lag_mat[:, k], color=col, label=label)
    ax1a.axhline(LAG_THEORY[k], color=col, ls='--', alpha=0.4)
    ax1b.plot(times / t_dyn, lag_mat[:, k] / lag_mat[0, k],
              color=col, label='r_%d%%' % int(frac*100))

ax1a.set_xlabel('time  [t_dyn]')
ax1a.set_ylabel('Lagrangian radius')
ax1a.set_title('absolute  (dashed = theory)')
ax1a.legend(fontsize=8)
ax1a.grid(True)

ax1b.axhline(1.0, color='k', ls='--', label='= 1')
ax1b.set_xlabel('time  [t_dyn]')
ax1b.set_ylabel('r / r(t=0)')
ax1b.set_title('normalised drift  (= 1 means stable)')
ax1b.legend(fontsize=8)
ax1b.grid(True)

plt.tight_layout()
plt.savefig("summary_lagrangian.pdf")
plt.close(fig1)
print("\nSaved: summary_lagrangian.pdf")


# ============================================================
#  FIGURE 2: VIRIAL RATIO AND VELOCITY DIAGNOSTICS vs TIME
# ============================================================

sv_theory = math.sqrt(G * M_tot / (2.0 * b))

fig2, (ax2a, ax2b, ax2c) = plt.subplots(1, 3, figsize=(14, 5))
fig2.suptitle('Velocity diagnostics  (N=%d, b=%.1f,  %.1f t_dyn)' % (N, b, times[-1]/t_dyn))

ax2a.plot(times / t_dyn, virial_ratio, 'b-', label='2K/|W|')
ax2a.axhline(1.0, color='k', ls='--', label='= 1')
ax2a.axhline(virial_mean, color='b', ls=':', label='mean = %.4f' % virial_mean)
ax2a.set_ylim(0.8, 1.2)
ax2a.set_xlabel('time  [t_dyn]')
ax2a.set_ylabel('2K/|W|')
ax2a.set_title('virial ratio')
ax2a.legend()
ax2a.grid(True)

ax2b.plot(times / t_dyn, sigma_v,  'b-',  label='sigma_v total')
ax2b.plot(times / t_dyn, sigma_vr, 'b--', label='sigma_vr')
ax2b.plot(times / t_dyn, sigma_vt, 'b:',  label='sigma_vt (1D)')
ax2b.axhline(sv_theory, color='k', ls='--', label='theory = %.4f' % sv_theory)
ax2b.set_xlabel('time  [t_dyn]')
ax2b.set_ylabel('velocity dispersion')
ax2b.set_title('velocity dispersion')
ax2b.legend()
ax2b.grid(True)

ax2c.plot(times / t_dyn, mean_vr_t, 'g-', label='<v_r>')
ax2c.axhline(0.0, color='k', ls='--', label='= 0')
ax2c.set_xlabel('time  [t_dyn]')
ax2c.set_ylabel('<v_r>')
ax2c.set_title('mean radial velocity  (should be ~0)')
ax2c.legend()
ax2c.grid(True)

plt.tight_layout()
plt.savefig("summary_velocities.pdf")
plt.close(fig2)
print("Saved: summary_velocities.pdf")


# ============================================================
#  FIGURE 3: DENSITY PROFILE — initial / mid / final
# ============================================================

t_mid_target = times[-1] / 2.0
pos_mid, vel_mid, phi_mid, t_mid_actual = get_mid_snapshot(
    filepath, N, t_mid_target, has_phi=HAS_PHI)
print(f"  Density mid snapshot at t = {t_mid_actual:.4f}")

r_theory = np.logspace(np.log10(b * 0.05), np.log10(b * 30), 300)
rho_th   = plummer_rho(r_theory, M_tot, b)

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
#  FIGURE 4: VELOCITY DISTRIBUTION P(q) — initial and final
# ============================================================

if phi_initial is not None and phi_last is not None:
    q_theory = np.linspace(0, 1, 300)
    p_theory = q_theory**2 * (1.0 - q_theory**2)**3.5
    p_theory /= np.trapezoid(p_theory, q_theory)

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(11, 5))
    fig4.suptitle('Velocity distribution P(q) vs theory  (m=5 polytrope DF)')

    for ax, label, vel_s, phi_s, t_s in [
            (ax4a, 't=0',                   vel_initial, phi_initial, times[0]),
            (ax4b, 't=%.2f' % times[-1],    vel_last,    phi_last,    times[-1])]:

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
    plt.close(fig4)
    print("Saved: summary_vel_dist.pdf")


# ============================================================
#  FIGURE 5: ANISOTROPY PROFILE beta(r) — initial and final
# ============================================================

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(11, 5))
fig5.suptitle('Anisotropy parameter beta(r)  (beta=0 means isotropic)')

for ax, label, pos_s, vel_s, t_s in [
        (ax5a, 't=0',                  pos_initial, vel_initial, times[0]),
        (ax5b, 't=%.2f' % times[-1],   pos_last,    vel_last,    times[-1])]:

    r_b, beta_b = anisotropy_profile(pos_s, vel_s)
    if len(r_b) > 0:
        ax.scatter(r_b, beta_b, color='blue', s=20)
        ax.plot(r_b, beta_b, 'b-', alpha=0.5)
    ax.axhline(0.0, color='k', ls='--', label='beta=0 (isotropic)')
    ax.axvline(b, color='orange', ls=':', label='r=b (softening)')
    ax.set_xscale('log')
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel('r')
    ax.set_ylabel('beta')
    ax.set_title(label)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("summary_anisotropy.pdf")
plt.close(fig5)
print("Saved: summary_anisotropy.pdf")


# ============================================================
#  FIGURE 6: SCALAR SUMMARY PANEL
# ============================================================

fig6, (ax6a, ax6b, ax6c) = plt.subplots(1, 3, figsize=(13, 5))
fig6.suptitle('Scalar stability metrics  (N=%d, b=%.1f, eps=0.012,  %.1f t_dyn)' % (
    N, b, times[-1]/t_dyn))

ax6a.plot(times / t_dyn, r_hm_t, 'b-', label='r_hm (sim)')
ax6a.axhline(r_hm_theory, color='r', ls='--', label='theory = %.4f' % r_hm_theory)
ax6a.set_xlabel('time  [t_dyn]')
ax6a.set_ylabel('r_hm')
ax6a.set_title('half-mass radius  (drift = %.2f%%)' % (100*r_hm_drift))
ax6a.legend()
ax6a.grid(True)

ax6b.plot(times / t_dyn, virial_ratio, 'b-', label='2K/|W|')
ax6b.axhline(1.0, color='k', ls='--', label='= 1')
ax6b.set_ylim(0.8, 1.2)
ax6b.set_xlabel('time  [t_dyn]')
ax6b.set_ylabel('2K/|W|')
ax6b.set_title('virial ratio  (mean = %.4f)' % virial_mean)
ax6b.legend()
ax6b.grid(True)

ax6c.set_xlabel('time  [t_dyn]')
if np.any(np.isfinite(energy_abs)) and energy_abs[0] > 0:
    e_err_t = np.abs(energy_abs - energy_abs[0]) / energy_abs[0]
    ax6c.semilogy(times / t_dyn, e_err_t, 'orange', label='|dE/E|')
    ax6c.axhline(E_ERR_THRESHOLD, color='r', ls='--', label='threshold')
    ax6c.legend()
ax6c.set_title('energy conservation  (max = %.2e)' % energy_err)
ax6c.grid(True)

plt.tight_layout()
plt.savefig("summary_scalars.pdf")
plt.close(fig6)
print("Saved: summary_scalars.pdf")


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
print(f"  Mean beta (final):   {beta_mean:.4f}  (0 = isotropic)")
print(f"  Max |Delta_E/E|:     {energy_err:.2e}  "
      f"({'PASS' if is_good else 'FAIL'})")
print("=" * 65)


# ============================================================
#  SERIALISE TO NPZ
# ============================================================

lag_keys = {f"lag_{int(f*100):02d}": lag_mat[:, k]
            for k, f in enumerate(LAG_FRACS)}

# ---- compute fixed-bin profiles for combined mode ----

# density profile at final snapshot (fixed log bins)
_cm_f   = np.mean(pos_last, axis=0)
_r_f    = np.linalg.norm(pos_last - _cm_f, axis=1)
_bins_d = np.logspace(np.log10(b * 0.05), np.log10(b * 30), 36)
_cnt, _ = np.histogram(_r_f, bins=_bins_d)
_vol    = (4.0/3.0) * np.pi * (_bins_d[1:]**3 - _bins_d[:-1]**3)
_rho    = np.where(_cnt > 0, _cnt * mass / _vol, np.nan)
_r_mid_d = np.sqrt(_bins_d[:-1] * _bins_d[1:])

# Jeans / beta profile at final snapshot (fixed log bins)
_bins_j  = np.logspace(-1.0, 1.0, 21)
_r_mid_j = np.sqrt(_bins_j[:-1] * _bins_j[1:])
_vel_c   = vel_last - np.mean(vel_last, axis=0)
_pos_c   = pos_last - _cm_f
_r_j     = np.linalg.norm(_pos_c, axis=1)
_rhat    = _pos_c / np.where(_r_j[:, None] == 0, 1e-10, _r_j[:, None])
_vr_j    = np.sum(_vel_c * _rhat, axis=1)
_vt2_j   = np.maximum(np.sum(_vel_c**2, axis=1) - _vr_j**2, 0.0)
_sigr    = np.full(20, np.nan)
_sigt    = np.full(20, np.nan)
_beta_j  = np.full(20, np.nan)
for _k in range(20):
    _m = (_r_j >= _bins_j[_k]) & (_r_j < _bins_j[_k+1])
    if _m.sum() > 5:
        _sigr[_k] = np.std(_vr_j[_m])
        _sigt[_k] = np.sqrt(np.mean(_vt2_j[_m]) / 2.0)
        if _sigr[_k] > 0:
            _beta_j[_k] = 1.0 - (_sigt[_k]**2 / _sigr[_k]**2)

# circular velocity at final snapshot
_r_sorted  = np.sort(_r_f)
_M_cum     = np.arange(1, N+1) * mass
_mask_vc   = _r_sorted > 0.01
_r_vc      = _r_sorted[_mask_vc]
_vcirc_sim = np.sqrt(G * _M_cum[_mask_vc] / _r_vc)

# velocity distribution P(q) at final snapshot (fixed bins)
if phi_last is not None:
    _vmag  = np.linalg.norm(vel_last, axis=1)
    _vesc  = np.sqrt(np.maximum(-2.0 * phi_last, 0.0))
    _q_sim = np.where(_vesc > 0, _vmag / _vesc, 1.0)
    _q_sim = _q_sim[_q_sim < 1.0]
    _bins_q   = np.linspace(0, 1, 41)
    _q_hist, _ = np.histogram(_q_sim, bins=_bins_q, density=True)
    _q_mid    = 0.5 * (_bins_q[:-1] + _bins_q[1:])
else:
    _q_hist = np.full(40, np.nan)
    _q_mid  = np.linspace(0, 1, 40)

# energy distribution at final snapshot
if phi_last is not None:
    _E_final = 0.5 * np.sum(vel_last**2, axis=1) + phi_last
else:
    _E_final = np.full(N, np.nan)

np.savez(
    "plummer_stats.npz",
    b             = np.array(b),
    t_dyn         = np.array(t_dyn),
    r_hm_theory   = np.array(r_hm_theory),
    times         = times,
    virial_ratio  = virial_ratio,
    sigma_v       = sigma_v,
    sigma_vr      = sigma_vr,
    sigma_vt      = sigma_vt,
    mean_vr       = mean_vr_t,
    energy_abs    = energy_abs,
    r_hm_drift    = np.array(r_hm_drift),
    sigma_v_drift = np.array(sv_drift),
    virial_mean   = np.array(virial_mean),
    virial_std    = np.array(virial_std),
    beta_mean     = np.array(beta_mean),
    energy_err    = np.array(energy_err),
    is_good       = np.array(is_good),
    # profiles for combined mode
    rho_r_mid     = _r_mid_d,
    rho_profile   = _rho,
    jeans_r_mid   = _r_mid_j,
    jeans_sigr    = _sigr,
    jeans_sigt    = _sigt,
    jeans_beta    = _beta_j,
    vcirc_r       = _r_vc,
    vcirc_sim     = _vcirc_sim,
    q_mid         = _q_mid,
    q_hist        = _q_hist,
    E_final       = _E_final,
    **lag_keys,
)
print("\nSaved: plummer_stats.npz")
print("Run 'python3 summary_runs.py --combined' from BASE_DIR for cross-run plots.")
