# === Pax-Dualon | UUCET | v5.2 (HUD + RGB toggle + Music presets; Brane demo ready) ===
# Panels: [A] φ mid-plane surface | [B] RGB Composite (DM=R, BM=G, DE=B) + HUD | [C] Cymatic Surface
#         [D] Topography + Contours | [E] Spectral Fingerprint | [F] Coupling Check (+ Entropy, CFL-ish, SW)
#         [G] BH Compactness (C_over vs. t)
# Notes:
#  - Panel B now overlays a HUD with energy fractions (DE/DM/BM), which match the printed values.
#  - RGB calibration can be switched between 'balanced' (per-channel autoscale) and 'truth' (global scale).
#  - One-click musical presets and a Brane pulse demo are included (call helpers BEFORE running the cell).
#  - All new features default OFF; paste-and-run preserves your legacy behavior.

%matplotlib inline
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from IPython.display import display

# ========= VERSION / RUNTIME =========
PAX_VERSION = "5.2"
N = 64                 # grid points per side
L = 40.0               # domain size
dx = L / N
dt = 0.025             # ~ dx^2/6 ≈ 0.0289 (safe but watch drivers)
STEPS = 4000
SAMPLE = 100           # sample metrics every SAMPLE steps

# ========= CORE 4 KNOBS (legacy-compatible) =========
w1, w2 = 1.0, 0.8      # base driver tones (dimensionless), used with Omega0 in driver_parametric
det = 0.2997           # frequency offset added to second tone (units of 1/time)
driver_scale = 0.3515  # amplitude for parametric driver

# ========= RGB calibration (view) =========
# 'balanced' = each channel autoscaled by its own percentile (prettier, not proportional)
# 'truth'    = one shared global scale so relative strengths match energy fractions
RGB_CAL_MODE = 'balanced'   # default; use the toggle widget to switch post-render
RGB_PERCENTILE = 95.0       # percentile for robust scaling

# ========= MUSIC LAYER (optional) =========
# Map musical intervals and tempo to (w1, w2, det, driver_scale) safely.
MUSIC_ENABLE        = False
MUSIC_TEMPERAMENT   = 'ET12'   # 'ET12' or 'PYTH'
MUSIC_INTERVAL      = +7       # semitones (e.g., +7 ≈ P5)
MUSIC_BPM           = None     # if set (e.g., 84), scales both tones proportionally
MUSIC_DET_RANGE     = (0.08, 0.35)   # keep det in this band (pre-tempo scaling)
MUSIC_KEEP_IMPRINT  = True     # retune driver_scale to preserve imprint strength when det changes

# ========= PHYSICS (PDRI + v4.7 merged) =========
Omega0 = 1.313
m_phi = 0.313
m_chi = 1.66
m_psi = 0.90
Gamma = 0.015
alphaBR = 0.005
gamma = 0.0313
gc = 2.0
gv = 0.005

A0 = 1.2
mA = 0.8
g = 4.0

B0 = 0.05
lorentz_gain = 0.01
m_spiral = 2
omega_spin = 0.90
fil_scale = 0.20
r_scale_frac = 0.35

# Layer toggles
USE_PARAMETRIC = True
USE_LORENTZ = True
USE_SPIRAL = True
USE_BANDS = True

# Band assist params
band_strength = 0.12
band_q_factor = 0.08           # still used, but we ensure >= ~1.5 dk below
resonant_bands = [1, 5]

# ========= BRANE MODE (orthogonal shock/standing waves, optional) =========
BRANE_ENABLE       = False      # toggle True to activate brane collision driver
BRANE_SIGMA        = L/10       # sheet thickness (Gaussian)
BRANE_LAMBDA_X     = L/3        # wavelength along x-sheet
BRANE_LAMBDA_Y     = L/3        # wavelength along y-sheet
BRANE_PHASE_X      = 0.0
BRANE_PHASE_Y      = 0.0
BRANE_GAIN         = 0.28       # amplitude for brane driver (kept moderate)
BRANE_USE_BPM      = True       # if MUSIC_BPM set, brane tones scale with tempo
# Time envelope for a single pulsed collision
BRANE_PULSE_CENTER = 0.6        # fraction of total run time for pulse center (0..1)
BRANE_PULSE_WIDTH  = 0.15       # fraction of total run time as std dev (Gaussian)
# Fusion gate → local boost where orthogonal waves overlap strongly
FUSION_THRESH_FRAC = 0.80       # threshold as percentile of instantaneous |Bx|, |By|
FUSION_GAIN_PHI    = 0.06       # add to φ acceleration where fused (gentle)
FUSION_DAMP_DROP   = 0.35       # reduce gamma locally for ψ where fused

# ========= GRID (precomputed) =========
x = np.linspace(0, L, N, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
cx, cy, cz = X - L/2, Y - L/2, Z - L/2
r = np.sqrt(cx**2 + cy**2 + cz**2)
th = np.arctan2(cy, cx)
r_scale = r_scale_frac * L
Rmax2 = (0.5*L)**2

kx = 2*np.pi*np.fft.fftfreq(N, dx)
ky = 2*np.pi*np.fft.fftfreq(N, dx)
kz = 2*np.pi*np.fft.fftfreq(N, dx)
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
K_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
dk = 2*np.pi / L   # one k-bin in magnitude

# Precompute brane Gaussians and wave numbers
if BRANE_ENABLE:
    Gx = np.exp(-(cy**2 + cz**2) / (2.0*BRANE_SIGMA**2))
    Gy = np.exp(-(cx**2 + cz**2) / (2.0*BRANE_SIGMA**2))
    kx0 = 2*np.pi / BRANE_LAMBDA_X
    ky0 = 2*np.pi / BRANE_LAMBDA_Y

# ========= NUMERICS =========
lap_weights = 1.0 / (dx*dx)
def lap3(F):
    return (np.roll(F, 1, 0) + np.roll(F, -1, 0) +
            np.roll(F, 1, 1) + np.roll(F, -1, 1) +
            np.roll(F, 1, 2) + np.roll(F, -1, 2) - 6.0*F) * lap_weights

# ========= MUSIC HELPERS =========
def _ratio_from_semitones(n, temperament='ET12'):
    n = int(n)
    if temperament.upper() == 'PYTH':
        tbl = {0:1/1,1:256/243,2:9/8,3:32/27,4:81/64,5:4/3,6:729/512,7:3/2,8:128/81,9:27/16,10:16/9,11:243/128,12:2/1}
        k = ((n % 12) + 12) % 12
        octs = (n - k)//12
        return tbl[k]*(2.0**octs)
    # Default 12-TET
    return 2.0**(n/12.0)

def _tau_coh(det_val, w1_val, w2_val):
    # Beat frequency between tone1 and (tone2 + det)
    wb = abs((w1_val*Omega0) - (w2_val*Omega0 + det_val))
    wb = max(wb, 1e-3)
    return 2*np.pi / wb

def _measure_tau_dyn_mid():
    # Mid-plane azimuthal speed scale as proxy (independent pre-run)
    v_circ = max(omega_spin * (L/6), 1e-3)
    r_h = L/6
    return (2*np.pi*r_h)/v_circ

def apply_music_adapter():
    global w1, w2, det, driver_scale
    if not MUSIC_ENABLE:
        return
    R = _ratio_from_semitones(MUSIC_INTERVAL, MUSIC_TEMPERAMENT)
    # Keep det within target band by nudging w2; aim (w2*Ω0 + det) ≈ R*(w1*Ω0)
    det_lo, det_hi = MUSIC_DET_RANGE
    target = R * (w1*Omega0)
    det_raw = target - (w2*Omega0)
    det_new = float(det_raw)
    w2_new = float(w2)
    if det_new < det_lo:
        w2_new = (target - det_lo)/Omega0
        det_new = det_lo
    elif det_new > det_hi:
        w2_new = (target - det_hi)/Omega0
        det_new = det_hi
    # Tempo scaling: scale both tones and det linearly with BPM/60
    if MUSIC_BPM is not None:
        pace = MUSIC_BPM/60.0
        w1 = w1*pace
        w2_new = w2_new*pace
        det_new = det_new*pace
    # Imprint preservation via coherence-time
    if MUSIC_KEEP_IMPRINT:
        tau_dyn = _measure_tau_dyn_mid()
        tau0 = _tau_coh(det, w1, w2)       # baseline
        taun = _tau_coh(det_new, w1, w2_new)
        eff0 = min(tau0/tau_dyn, 1.0)
        effn = max(min(taun/tau_dyn, 1.0), 1e-6)
        I = driver_scale * eff0
        driver_scale = float(np.clip(I/effn, 0.05, 1.00))
    # Commit w2, det
    w2 = w2_new
    det = det_new
    print(f"[music] temperament={MUSIC_TEMPERAMENT} semitones={MUSIC_INTERVAL} → w1={w1:.3f}, w2={w2:.3f}, det={det:.3f}, driver_scale={driver_scale:.3f}")

# One-click musical presets (call before running the cell)
def set_music_preset(name='ET12_fifth_84bpm'):
    global MUSIC_ENABLE, MUSIC_TEMPERAMENT, MUSIC_INTERVAL, MUSIC_BPM
    presets = {
        'ET12_fifth_84bpm':  ('ET12', +7, 84),
        'PYTH_major_third_60bpm': ('PYTH', +4, 60),
        'octave_96bpm':      ('ET12', +12, 96),
        'off':               (None, None, None),
    }
    if name not in presets:
        print(f"[music] unknown preset '{name}'. Choices: {list(presets.keys())}")
        return
    sys, semi, bpm = presets[name]
    if sys is None:
        MUSIC_ENABLE = False
        print("[music] disabled (preset 'off'). Re-run the cell to take effect.")
        return
    MUSIC_ENABLE = True
    MUSIC_TEMPERAMENT = sys
    MUSIC_INTERVAL = semi
    MUSIC_BPM = bpm
    print(f"[music] preset '{name}' staged. It will apply when this cell runs. (sys={sys}, semitones={semi}, BPM={bpm})")

# Brane pulse quick preset (call before running the cell)
def set_brane_pulse_demo():
    global BRANE_ENABLE, BRANE_GAIN, BRANE_PULSE_CENTER, BRANE_PULSE_WIDTH
    BRANE_ENABLE = True
    BRANE_GAIN = 0.28
    BRANE_PULSE_CENTER = 0.6
    BRANE_PULSE_WIDTH = 0.15
    print("[brane] Pulse demo enabled (orthogonal standing-wave lattice). Re-run the cell to take effect.")

# ========= DRIVERS =========
def driver_parametric(t, phi):
    if not USE_PARAMETRIC: return 0.0
    # Two-tone with offset det; Ω0 scales dimensionless w1,w2 to angular frequency
    osc = np.cos(w1*Omega0*t) + np.cos(w2*Omega0*t + det*t)
    return driver_scale * 1.66 * osc * (1.0 - 0.25*np.tanh(phi))

def driver_spiral(t):
    if not USE_SPIRAL: return 0.0
    return fil_scale * np.exp(-r/r_scale) * np.sin(m_spiral*th) * np.cos(omega_spin*t)

def rotation_bowl(t):
    if not USE_SPIRAL: return 0.0
    return 0.5*(omega_spin**2)*(1.0 - (cx**2 + cy**2)/Rmax2)

def lorentz_sweep(phi, vx, vy, vz):
    if not USE_LORENTZ: return (0.0, 0.0, 0.0)
    dpx = (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / (2*dx)
    dpy = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / (2*dx)
    dpz = (np.roll(phi, -1, 2) - np.roll(phi, 1, 2)) / (2*dx)
    Ex, Ey, Ez = -dpx, -dpy, -dpz
    Sx, Sy, Sz = Ey*B0, -Ex*B0, 0.0*Ez
    return (lorentz_gain*Sx, lorentz_gain*Sy, lorentz_gain*Sz)

def band_resonance_filter(field, t, bands):
    if not USE_BANDS: return 0.0
    Fk = np.fft.fftn(field)
    out = np.zeros_like(Fk)
    modulation = np.cos(Omega0*t)
    for n in bands:
        k0 = np.sqrt(max(0.0, (n*Omega0)**2 - m_psi**2))
        if k0 == 0.0:
            continue
        # ensure bandwidth covers >= ~1.5 k-bins
        bw = max(1.5*dk, band_q_factor * max(k0, 1e-6))
        mask = np.exp(-((K_mag - k0)/bw)**2)
        gain = band_strength * (1.2 if n == 5 else 1.0)
        out += Fk * mask * gain * modulation
    return np.real(np.fft.ifftn(out))

def brane_driver_fields(t):
    """
    Orthogonal brane sheets: standing-wave friendly pairs along x and y with Gaussian thickness.
    Returns Bx, By, env.
    """
    if not BRANE_ENABLE:
        return 0.0, 0.0, 0.0
    # Envelope: centered Gaussian pulse (single event); width as fraction of total time
    Ttot = STEPS*dt
    t0 = BRANE_PULSE_CENTER*Ttot
    sig_t = max(1e-6, BRANE_PULSE_WIDTH*Ttot)
    env = np.exp(-0.5*((t - t0)/sig_t)**2)
    # Frequencies: tie to parametric tones; respect BPM scaling if requested
    w1p = w1*Omega0
    w2p = (w2*Omega0)
    if MUSIC_BPM is not None and BRANE_USE_BPM:
        pace = MUSIC_BPM/60.0
        w1p *= pace; w2p *= pace
    # Forward and backward components for standing waves
    Bx = Gx * (np.cos(kx0*cx - w1p*t + BRANE_PHASE_X) + np.cos(kx0*cx + w1p*t + BRANE_PHASE_X))
    By = Gy * (np.cos(ky0*cy - w2p*t + BRANE_PHASE_Y) + np.cos(ky0*cy + w2p*t + BRANE_PHASE_Y))
    return Bx, By, env

# ========= INITIAL FIELDS =========
rng = np.random.default_rng(42)
phi = 0.02 * rng.standard_normal((N, N, N))
chi = 0.02 * rng.standard_normal((N, N, N))
psi = 0.01 * rng.standard_normal((N, N, N))
Ax = 1e-6 * rng.standard_normal((N, N, N))
Ay = 1e-6 * rng.standard_normal((N, N, N))
Az = 1e-6 * rng.standard_normal((N, N, N))

# velocities & time-derivatives
vx = np.zeros_like(phi); vy = np.zeros_like(phi); vz = np.zeros_like(phi)
phi_dot = np.zeros_like(phi); chi_dot = np.zeros_like(chi); psi_dot = np.zeros_like(psi)
Vx = np.zeros_like(Ax); Vy = np.zeros_like(Ay); Vz = np.zeros_like(Az)

# Apply music adapter (if enabled) before run
apply_music_adapter()

# ========= METRICS & HELPERS =========
def f_void(F, thresh=0.35):
    a = np.abs(F); m = a.max() + 1e-12
    return float(np.mean(a < thresh*m))

def f_visible(F, thresh=0.25):
    a = np.abs(F); m = a.max() + 1e-12
    return float(np.mean(a > thresh*m))

def _grad2(F):
    return ((np.roll(F, -1, 0) - F) / dx)**2 + ((np.roll(F, -1, 1) - F) / dx)**2 + ((np.roll(F, -1, 2) - F) / dx)**2

def energy_proxy(phi, chi, psi):
    pot = 0.5 * (m_phi**2 * phi**2 + m_chi**2 * chi**2 + m_psi**2 * psi**2)
    return float(np.mean(_grad2(phi) + _grad2(chi) + _grad2(psi) + pot))

def energy_fractions(phi, chi, psi):
    """Fractions by per-field energy contributions (sum to 1)."""
    Ephi = np.mean(_grad2(phi) + 0.5*m_phi**2 * phi**2)
    Echi = np.mean(_grad2(chi) + 0.5*m_chi**2 * chi**2)
    Epsi = np.mean(_grad2(psi) + 0.5*m_psi**2 * psi**2)
    Etot = max(Ephi + Echi + Epsi, 1e-30)
    return (Ephi/Etot, Echi/Etot, Epsi/Etot)

def curl_components(Ax, Ay, Az, dx):
    d = 2*dx
    dAy_dz = (np.roll(Ay, -1, 2) - np.roll(Ay, 1, 2)) / d
    dAz_dy = (np.roll(Az, -1, 1) - np.roll(Az, 1, 1)) / d
    dAz_dx = (np.roll(Az, -1, 0) - np.roll(Az, 1, 0)) / d
    dAx_dz = (np.roll(Ax, -1, 2) - np.roll(Ax, 1, 2)) / d
    dAx_dy = (np.roll(Ax, -1, 1) - np.roll(Ax, 1, 1)) / d
    dAy_dx = (np.roll(Ay, -1, 0) - np.roll(Ay, 1, 0)) / d
    return dAy_dz - dAz_dy, dAz_dx - dAx_dz, dAx_dy - dAy_dx

def vec_energy(Ax, Ay, Az, Vx, Vy, Vz, t):
    Cx, Cy, Cz = curl_components(Ax, Ay, Az, dx)
    phi_t = A0 * np.cos(m_phi * t)
    mass_term = mA**2 + g * (phi_t**2)
    v2 = Vx*Vx + Vy*Vy + Vz*Vz
    curl2 = Cx*Cx + Cy*Cy + Cz*Cz
    A2 = Ax*Ax + Ay*Ay + Az*Az
    return float(np.mean(0.5 * (v2 + curl2 + mass_term * A2)))

def angmom_vector(Ax, Ay, Az, Vx, Vy, Vz, dx):
    Cx, Cy, Cz = curl_components(Ax, Ay, Az, dx)
    Sx = Vy*Cz - Vz*Cy
    Sy = Vz*Cx - Vx*Cz
    Sz = Vx*Cy - Vy*Cx
    N = Ax.shape[0]
    xs = (np.arange(N) + 0.5) - (N/2.0)
    Xg, Yg, Zg = np.meshgrid(xs, xs, xs, indexing='ij')
    rx, ry, rz = Xg*dx, Yg*dx, Zg*dx
    lx = ry*Sz - rz*Sy
    ly = rz*Sx - rx*Sz
    lz = rx*Sy - ry*Sx
    return np.sqrt(lx*lx + ly*ly + lz*lz)

def energy_density_field(phi, chi, psi, dx):
    pot = 0.5*(m_phi**2*phi**2 + m_chi**2*chi**2 + m_psi**2*psi**2)
    return _grad2(phi) + _grad2(chi) + _grad2(psi) + pot

def bh_compactness_nd(Efld, Ldens, dx, qE=0.99, qL=0.995):
    thrE = np.quantile(Efld, qE)
    thrL = np.quantile(Ldens, qL)
    ov = (Efld >= thrE) & (Ldens >= thrL)
    if not ov.any():
        return 0.0
    V = ov.sum() * (dx**3)
    R_eff = (3.0*V/(4.0*np.pi))**(1.0/3.0)
    Mproxy = Efld[ov].sum() * (dx**3)
    Rs_nd = 2.0 * Mproxy      # absorb G/c^2 into units
    return float(Rs_nd / max(R_eff, 1e-12))

def spectral_entropy(field2d):
    F = np.fft.fftshift(np.fft.fft2(field2d))
    P = np.abs(F)**2
    P = P / (P.sum() + 1e-16)
    P = np.clip(P, 1e-16, 1.0)
    return float(-(P * np.log(P)).sum())

def soft_clip(F, z=6.0):
    s = np.std(F); m = np.mean(F)
    if s <= 0:
        return F
    lim = z*s
    return np.clip(F, m - lim, m + lim)

# ========= TIME LOOP =========
times, E_hist, V_hist, Vis_hist, L_hist, C_hist = [], [], [], [], [], []
ENT_hist, CFL_hist = [], []
SW_hist = []   # standing-wave score over time (mid-plane φ)

# Optional brane prealloc (safe defaults when disabled)
Bx = By = 0.0
env_b = 0.0
Fgate = None

t = 0.0
for step in range(1, STEPS + 1):
    # Optional brane fields
    if BRANE_ENABLE:
        Bx, By, env_b = brane_driver_fields(t)
        # Fusion gate thresholds (percentiles) computed on-the-fly to remain scale-free
        aBx = np.abs(Bx); aBy = np.abs(By)
        thrx = np.quantile(aBx, FUSION_THRESH_FRAC)
        thry = np.quantile(aBy, FUSION_THRESH_FRAC)
        Fgate = (aBx >= thrx) & (aBy >= thry)
    else:
        Bx = By = env_b = 0.0
        Fgate = None

    # φ evolution
    a_phi = lap3(phi) - Gamma * phi - alphaBR * (chi**2) * phi
    if USE_PARAMETRIC: a_phi += driver_parametric(t, phi)
    if USE_SPIRAL: a_phi += driver_spiral(t) + rotation_bowl(t)
    if BRANE_ENABLE:
        a_phi += BRANE_GAIN * env_b * (Bx + By)
        if Fgate is not None and np.any(Fgate):
            a_phi = a_phi + (FUSION_GAIN_PHI * Fgate.astype(phi.dtype))

    if USE_LORENTZ:
        Sx, Sy, Sz = lorentz_sweep(phi, vx, vy, vz)
        vx = 0.98 * vx + Sx * dt
        vy = 0.98 * vy + Sy * dt
        vz = 0.98 * vz + Sz * dt
        a_phi += (-vx * (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / (2*dx)
                 - vy * (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / (2*dx)
                 - vz * (np.roll(phi, -1, 2) - np.roll(phi, 1, 2)) / (2*dx)) * dt
    a_phi += gv * np.tanh(-phi) - gc * (0.25 * np.tanh(phi))
    phi_dot = soft_clip(phi_dot + a_phi * dt)
    phi = soft_clip(phi + phi_dot * dt)

    # χ evolution
    a_chi = lap3(chi) - gamma * chi - 0.25 * alphaBR * (phi**2) * chi
    chi_dot = soft_clip(chi_dot + a_chi * dt)
    chi = soft_clip(chi + chi_dot * dt)

    # ψ evolution
    # Local damping relief in fusion regions to emulate superposition thickness/cooling
    if BRANE_ENABLE and (Fgate is not None) and np.any(Fgate):
        gamma_local = gamma * (1.0 - FUSION_DAMP_DROP*Fgate.astype(psi.dtype))
    else:
        gamma_local = gamma
    a_psi = lap3(psi) - 0.5 * gamma_local * psi - 0.1 * alphaBR * (phi**2 + chi**2) * psi
    if USE_BANDS: a_psi += band_resonance_filter(psi, t, [1])  # start with n=1
    psi_dot = soft_clip(psi_dot + a_psi * dt)
    psi = soft_clip(psi + psi_dot * dt)

    # Ax/Ay/Az evolution
    phi_t = A0 * np.cos(m_phi * t)
    mass_term = mA**2 + g * (phi_t**2)
    aAx = lap3(Ax) - mass_term * Ax - Gamma * Vx
    aAy = lap3(Ay) - mass_term * Ay - Gamma * Vy
    aAz = lap3(Az) - mass_term * Az - Gamma * Vz
    Vx = soft_clip(Vx + aAx * dt); Ax = soft_clip(Ax + Vx * dt)
    Vy = soft_clip(Vy + aAy * dt); Ay = soft_clip(Ay + Vy * dt)
    Vz = soft_clip(Vz + aAz * dt); Az = soft_clip(Az + Vz * dt)

    # CFL-ish monitor (very rough): compare advective speed scale to grid/time
    vmax = float(np.max(np.sqrt(vx*vx + vy*vy + vz*vz)))
    CFL_hist.append(vmax * dt / max(dx, 1e-12))

    # Metrics
    if (step % SAMPLE == 0) or (step == STEPS):
        times.append(t)
        E_hist.append(float(energy_proxy(phi, chi, psi) + vec_energy(Ax, Ay, Az, Vx, Vy, Vz, t)))
        V_hist.append(float(f_void(phi)))
        Vis_hist.append(float(f_visible(psi)))
        Ldens = angmom_vector(Ax, Ay, Az, Vx, Vy, Vz, dx)
        L_hist.append(float(Ldens.mean()))
        Efld = energy_density_field(phi, chi, psi, dx)
        C_hist.append(float(bh_compactness_nd(Efld, Ldens, dx)))
        # mid-plane entropy (structure vs. order)
        ENT_hist.append(float(spectral_entropy(psi[:, :, N//2])))
        # standing-wave score on φ mid-plane (power along axes vs total)
        phi_mid_tmp = phi[:, :, N//2]
        F2 = np.abs(np.fft.fft2(phi_mid_tmp))**2
        axis_power = np.mean(F2[N//2, :]) + np.mean(F2[:, N//2])
        SW_hist.append(float(axis_power / (np.mean(F2) + 1e-12)))

    t += dt

# ========= FRACTIONS (energy-based; sum to 1) =========
f_DE, f_DM, f_BM = energy_fractions(phi, chi, psi)

# ========= PLOTS (7 PANELS, 3-WIDE) =========
mid = N // 2
phi_mid, chi_mid, psi_mid = phi[:, :, mid], chi[:, :, mid], psi[:, :, mid]
A_mid = np.sqrt(Ax[:, :, mid]**2 + Ay[:, :, mid]**2 + Az[:, :, mid]**2)

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 3)

# [A] φ mid-plane surface (single-cmap surface)
axA = fig.add_subplot(gs[0, 0], projection='3d')
surf = axA.plot_surface(X[:, :, mid], Y[:, :, mid], phi_mid, cmap='RdBu', rstride=1, cstride=1, alpha=0.85)
axA.set_title(f"A: φ mid-plane surface (v{PAX_VERSION})")
axA.set_xlabel("X"); axA.set_ylabel("Y"); axA.set_zlabel("φ")

# [B] RGB Composite (DM=R, BM=G, DE=B) with calibration modes + HUD
def _qscale(arr, p=95):
    return np.percentile(np.abs(arr), p)

def build_rgb_images(phi, chi, psi, phi_mid, chi_mid, psi_mid, p=95.0):
    # truth: one shared scale from full-volume fields for fair relative intensity
    S_global = max(_qscale(phi, p), _qscale(chi, p), _qscale(psi, p), 1e-12)
    r_truth = np.clip(np.abs(chi_mid) / S_global, 0, 1)
    g_truth = np.clip(np.abs(psi_mid) / S_global, 0, 1)
    b_truth = np.clip(np.abs(phi_mid) / S_global, 0, 1)
    rgb_truth = np.stack([r_truth, g_truth, b_truth], axis=-1)

    # balanced: per-channel scaling on the mid-plane
    r_bal = np.clip(np.abs(chi_mid) / max(_qscale(chi_mid, p), 1e-12), 0, 1)
    g_bal = np.clip(np.abs(psi_mid) / max(_qscale(psi_mid, p), 1e-12), 0, 1)
    b_bal = np.clip(np.abs(phi_mid) / max(_qscale(phi_mid, p), 1e-12), 0, 1)
    rgb_bal = np.stack([r_bal, g_bal, b_bal], axis=-1)

    return rgb_truth, rgb_bal

rgb_truth_img, rgb_bal_img = build_rgb_images(phi, chi, psi, phi_mid, chi_mid, psi_mid, RGB_PERCENTILE)
rgb_initial = rgb_bal_img if RGB_CAL_MODE == 'balanced' else rgb_truth_img
mode_label = RGB_CAL_MODE

axB = fig.add_subplot(gs[0, 1])
imB = axB.imshow(np.clip(rgb_initial, 0, 1), origin='lower',
                 extent=[-L/2, L/2, -L/2, L/2], aspect='equal')
axB.set_title(f"B: RGB Composite (DM=R, BM=G, DE=B) — {mode_label}")
axB.axis('off')

# Panel B HUD: energy fractions (match printed values)
hud_text = f"DE={f_DE:.3f}  DM={f_DM:.3f}  BM={f_BM:.3f}  | sum={f_DE+f_DM+f_BM:.3f}"
axB.text(0.98, 0.02, hud_text, transform=axB.transAxes,
         ha='right', va='bottom', fontsize=9, color='w',
         bbox=dict(boxstyle='round,pad=0.2', fc=(0,0,0,0.35), ec='none'))

# [C] Cymatic Surface (|A|)
axC = fig.add_subplot(gs[0, 2], projection='3d')
surf = axC.plot_surface(X[:, :, mid], Y[:, :, mid], A_mid, cmap='viridis', rstride=1, cstride=1, alpha=0.85)
axC.set_title("C: Cymatic Surface (|A|)")
axC.set_xlabel("X"); axC.set_ylabel("Y"); axC.set_zlabel("|A|")

# [D] Topography + Contours
axD = fig.add_subplot(gs[1, 0])
phi_mid_norm = phi_mid / (np.max(np.abs(phi_mid)) + 1e-12)
axD.imshow(phi_mid_norm.T, origin='lower', extent=[-L/2, L/2, -L/2, L/2], aspect='equal', cmap='terrain', alpha=0.95)
axD.contour(cx[:, :, mid], cy[:, :, mid], phi_mid_norm.T, levels=16, colors='k', linewidths=1.0, alpha=0.9)
axD.set_title("D: Topography φ + Contours")

# [E] Spectral Fingerprint P(k)
axE = fig.add_subplot(gs[1, 1])
phi_k = np.abs(np.fft.fftn(phi)) / phi.size
k_bins = np.linspace(0, np.max(K_mag), 50)
pk = [np.mean(phi_k[(K_mag >= k_bins[i]) & (K_mag < k_bins[i+1])]) for i in range(len(k_bins)-1)]
axE.plot(0.5 * (k_bins[1:] + k_bins[:-1]), pk)
axE.set_title("E: Spectral Fingerprint P(k)")
axE.set_xlabel("k"); axE.set_ylabel("Power")

# [F] Coupling Check (Energy, <|ℓ|>, Entropy, CFL-ish [+ SW score]))
axF = fig.add_subplot(gs[1, 2])
times_arr = np.array(times)
axF.plot(times_arr, np.array(E_hist), label='Energy')
axF.plot(times_arr, np.array(L_hist), label='Mean |ℓ|')
axF.plot(times_arr, np.array(ENT_hist), label='Spectral Entropy (ψ_mid)')
axF.plot(np.linspace(times_arr[0], times_arr[-1], len(CFL_hist))[::SAMPLE], np.array(CFL_hist)[::SAMPLE], label='CFL-ish')
if SW_hist:
    axF.plot(times_arr, np.array(SW_hist), label='SW score (standing-wave)')
axF.set_title("F: Coupling Check (+ Entropy, CFL-ish, SW)")
axF.set_xlabel("t"); axF.set_ylabel("Value"); axF.legend()

# [G] BH Compactness
axG = fig.add_subplot(gs[2, :])
C_arr = np.array(C_hist)
axG.plot(times_arr, C_arr, label='C_over')
axG.axhline(0.1, color='k', linestyle='--', alpha=0.5, label='C=0.1 (collapse)')
axG.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='C=1.0 (horizon)')
axG.set_title("G: BH Compactness (C_over)")
axG.set_xlabel("t"); axG.set_ylabel("C_over"); axG.legend()

plt.tight_layout()
plt.show()

# ========= WIDGET: Panel B RGB toggle (balanced <-> truth) =========
try:
    import ipywidgets as W
    from ipywidgets import HBox, VBox, ToggleButtons, Output, Layout, HTML

    outB = Output(layout=Layout(border='1px solid #ccc'))
    mode_widget = ToggleButtons(
        options=[('Balanced (per-channel)', 'balanced'), ('Truth (global scale)', 'truth')],
        value=RGB_CAL_MODE,
        description='RGB:',
        button_style='',
        layout=Layout(width='420px')
    )

    def _render_panelB(mode):
        # Choose image
        img = rgb_bal_img if mode == 'balanced' else rgb_truth_img
        with outB:
            outB.clear_output(wait=True)
            fig2, ax2 = plt.subplots(1,1, figsize=(5.2,5.2), dpi=110)
            ax2.imshow(np.clip(img,0,1), origin='lower', extent=[-L/2, L/2, -L/2, L/2], aspect='equal')
            ax2.set_title(f"Panel B — {mode}")
            ax2.axis('off')
            # HUD
            hud = f"DE={f_DE:.3f}  DM={f_DM:.3f}  BM={f_BM:.3f}  | sum={f_DE+f_DM+f_BM:.3f}"
            ax2.text(0.98, 0.02, hud, transform=ax2.transAxes,
                     ha='right', va='bottom', fontsize=9, color='w',
                     bbox=dict(boxstyle='round,pad=0.2', fc=(0,0,0,0.35), ec='none'))
            display(fig2)
            plt.close(fig2)

    def _on_mode_change(change):
        if change['name'] == 'value':
            _render_panelB(change['new'])

    mode_widget.observe(_on_mode_change, names='value')
    # Initial render
    _render_panelB(mode_widget.value)
    display(VBox([HTML("<b>Panel B RGB Toggle</b>"), mode_widget, outB]))
except Exception as e:
    print("[widgets] ipywidgets not available. Install with `%pip install ipywidgets` to enable the Panel B RGB toggle.")
    print("          You can still switch modes by changing RGB_CAL_MODE at the top and re-running the cell.")

# ========= PRINTS =========
sum_chk = f_DE + f_DM + f_BM
print(f"[Fractions (energy-based)] DE={f_DE:.4f}  DM={f_DM:.4f}  BM={f_BM:.4f}  | sum≈{sum_chk:.4f}")
print(f"[Config] v{PAX_VERSION} | steps={STEPS} | N={N} | dt={dt} | RGB mode={RGB_CAL_MODE} | MUSIC_ENABLE={MUSIC_ENABLE} | BRANE_ENABLE={BRANE_ENABLE}")
if MUSIC_ENABLE:
    print(f"[Music] temperament={MUSIC_TEMPERAMENT} interval={MUSIC_INTERVAL} semitones | BPM={MUSIC_BPM} | det band={MUSIC_DET_RANGE}")
if BRANE_ENABLE:
    print(f"[Brane] σ={BRANE_SIGMA:.3f} λx={BRANE_LAMBDA_X:.2f} λy={BRANE_LAMBDA_Y:.2f} gain={BRANE_GAIN:.3f}")

# ========= HOW TO (quick reference) =========
print("\n[HowTo]")
print(" - To use a musical preset: set_music_preset('ET12_fifth_84bpm')  # or 'PYTH_major_third_60bpm', 'octave_96bpm', 'off'")
print("   Then re-run this cell so it applies before the run.")
print(" - To demo the brane pulse (orthogonal standing waves): set_brane_pulse_demo(); then re-run this cell.")
print(" - To switch RGB mode interactively, use the widget above; or set RGB_CAL_MODE at the top and re-run.")
