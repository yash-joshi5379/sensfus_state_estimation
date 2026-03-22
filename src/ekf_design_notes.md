# EKF Design Notes — myEKF_ca.m

## State Vector
`X = [x; y; theta; vx; vy; omega; ax; ay; b_omega]` (9×1)

Constant-acceleration kinematic model. Heading `theta` and angular rate `omega` are decoupled
from position/velocity in the prediction step — heading only changes via the gyro update.

`b_omega` is a gyro bias state (random walk) observable during stationary phases via the
zero-rotation pseudo-measurement. Frozen during motion (`Q(9,9) = 0` when not stationary)
to prevent spurious drift when there is no correction signal.

---

## Arena Geometry

**Walls at x = ±1.22 m, y = ±1.22 m** (total arena 2.44 m × 2.44 m, max diagonal ≈ 3.45 m).

> **Critical:** `Lx = Ly = 2.44` was used for a long time by mistake, placing walls at ±2.44 m
> (a 4.88×4.88 m arena). Every ToF prediction was ~2× too large, causing near-100% chi² gate
> rejection and open-loop position divergence. Fixed to `Lx = Ly = 1.22`.

---

## Sensor Usage

### Accelerometer — `acc(2)`, `acc(3)`
**Used:** Body-frame ax, ay measurement, every step.
IMU update observes `[ax_body; ay_body]` = world-frame acceleration rotated into body frame.
This constrains the `ax`, `ay` world-frame states which drive position prediction.

**Not used for heading:** Gravity is perpendicular to the robot's horizontal plane.
The horizontal accelerometer axes carry zero gravity component regardless of yaw —
accelerometers cannot observe yaw rotation.

**During `fast_spin`:** Acc noise inflated from R = (0.5)² to (5.0)² m²/s⁴.
Reason: centripetal acceleration during fast rotation is a real physical signal in body frame
but projects to world-frame `ax`, `ay` and drives position drift. Inflating acc noise
makes the filter ignore the acceleration measurement during spin. `fast_spin` is detected
using raw `gyro_z` (not the estimated `omega` state, which starts at 0 even if the robot is
already spinning at initialisation).

**Process noise:** `Q(7,7) = Q(8,8) = (0.15)²` — reduced from `(0.30)²` to limit acc noise
bleeding into position via P off-diagonal covariance terms.

---

### Gyroscope — `gyro(1)`
**Used:** Yaw rate measurement every step, low noise R = (0.02)² rad²/s².
Primary heading sensor — integrates to `theta` through the `omega` state.

**Also used for:**
- `fast_spin` detection: `|gyro_z| > 0.5 rad/s`
- Magnetometer seed gate: `|gyro_z| < 0.15 rad/s` at step 1 (motors-off check)
- Stationary detection: `|gyro_z| < 0.10 rad/s` (combined with acc threshold)

**Known issue — motor EMI bias shift:**
Gyro bias calibrated from stationary data (calib2_straight, first ~12 000 samples) via
`bias_calc.m` gives `gyro_x_bias = -0.0112 rad/s`. During motor operation the effective bias
shifts (motor EMI), causing heading drift over long runs.

`gyro_scale = 1.1` corrects ~10% over-integration observed in task datasets.
Note: at 1.02 the straight dataset was near-perfect but task datasets under-integrated;
1.1 is the best single constant across all datasets.

---

### Magnetometer — `mag(2)`, `mag(3)`
**Used:** One-time heading seed at **step 1 only**, gated on `|gyro_z| < 0.15` (motors off).

Formula:
```
theta_0 = wrapToPi(atan2(mag(3) - mag_y_bias, mag(2) - mag_x_bias) + mag_declination_static)
```
`mag_declination_static = -1.4245 rad` — calibrated from calib2_straight stationary frames
(motors off, low EMI). This differs from `mag_declination = 1.168 rad` which was calibrated
from the rotation dataset with motors running and has ~180° of EMI baked in.

After seed, `P(3,3)` is tightened to `deg2rad(10)²`.

For test purposes (`run_ekf_test.m`), a `fake_mag` vector encoding the GT heading and a
`fake_gyro` with `gyro(1) = gyro_x_bias` (forcing `gyro_z = 0`) are injected at step 1
to guarantee the seed fires regardless of motor state.

**Not used for ongoing updates.**
Reason: motor EMI causes ~180° heading interference during operation. The effective
declination varies with motor speed and load, so there is no stable calibration constant
to use during motion.

**Tried: slow-motion mag updates** (`|gyro_z| < 0.3`, `|v| < 0.15 m/s`, `|acc| < 0.3 m/s²`).
Result: caused heading jumps at the start of datasets (motor ramp-up has intermediate EMI,
neither static nor full-speed) and continued drift elsewhere. The `mag_declination` baseline
is wrong for slow-speed operation. Removed.

---

### Gyro Bias State — `b_omega` (state 9)
**Used:** Tracks residual gyro bias above the static `gyro_x_bias` calibration.

Observable via the gyro measurement model: `h_gyro = omega + b_omega`, so `H_imu(3,9) = 1`.

**Zero-rotation pseudo-measurement:** When stationary (`|gyro_z| < 0.10`, `|acc_body| < 0.15`),
a pseudo-measurement `omega = 0` with `R_zr = (0.01)²` is applied. Any gyro reading above
noise is therefore attributed to bias — this drives `b_omega` estimation during stops.

**Frozen during motion:** `Q(9,9) = 0` when not stationary, so `P(9,9)` does not grow and
the Kalman gain for `b_omega` stays near zero. Prevents spurious bias drift during motion
where there is no correction signal. Only `Q(9,9) = (3e-3)²` (random walk) during stationary
phases to allow slow adaptation to EMI changes at rest.

---

### ToF Sensors — ToF1 (right), ToF2 (forward), ToF3 (left)
**Used:** Range-to-wall position updates at 10 Hz (every 20th step, `tof_update_freq = 20`).

Ray geometry:
```
tof_phi = [-pi/2; 0; pi/2]   % right, forward, left (body frame)
```
Sensor world position computed from robot position + offset rotated by heading.
Range predicted via ray–wall intersection with analytical Jacobians `[dh/dsx, dh/dsy]`.

**Chi² gate:** `chi2_thresh = 4.0` (tightened from 6.63).
Rejects outliers from wall holes, corners, and reflections. Tighter gate reduces sawtooth
position artifacts caused by borderline outliers pulling position in transient wrong directions.

**R_tof = (0.15)²** — increased from (0.05)² to reduce the magnitude of each individual ToF
correction. Root cause of position sawtooth: slightly-off heading estimate + noisy ToF
readings cause consecutive updates to pull position in slightly different directions.
Higher R_tof makes the filter lean more on the kinematic model between updates.

**During `fast_spin`:** Position process noise inflated (`Q(1,1) = Q(2,2) = (0.15)²`)
so that `P` grows faster → `S_tof` grows → chi² gate naturally widens.
Reason: centripetal acc can drift position estimate slightly before the gate activates;
wider gate allows valid ToF measurements through to pull position back.

**Not used for heading (`H_tof(3) = 0`, `K_tof(3) = 0`).**
The full analytical heading Jacobian `dh/dtheta` was tested at various gain levels (100%,
5%). Result: violent heading spikes (full ±π oscillations) across all datasets.
Root cause: `dh/dtheta` can be several m/rad at oblique wall angles. With even moderate
R_tof, any small position residual produces a gain `K_tof(3)` large enough to rotate heading
by radians per update. Zeroing both `H_tof(3)` and `K_tof(3)` (both must be zeroed together
to maintain `P` symmetry).

---

## Heading Drift — Known Limitation

With no ongoing heading correction, `theta` drifts with gyro bias error:

| Dataset | Observed drift | Source |
|---------|---------------|--------|
| straight (70 s) | Small (~0.03 rad) | Good static bias calibration; minimal rotation |
| task2_1, task2_2 | ~0.2 rad lag during rotation | Motor EMI shifts effective gyro bias |

**Options investigated and rejected:**
1. `H_tof(3)` — causes ±π heading spikes (see ToF section above)
2. Slow-motion mag updates — variable EMI baseline, causes jumps (see Mag section)
3. Accelerometer for heading — physically impossible (gravity ⊥ yaw axis)
4. Dynamic b_omega during motion — no correction signal available; causes spurious drift

**b_omega state** partially addresses this during stationary phases but cannot correct
drift accumulated during motor operation.

---

## GT Convention Note

The GT heading from the motion capture system is offset by **+π** from the robot's body-frame
heading (consequence of the board being upended with a 180° axis flip). This affects visual
comparison only — the EKF operates entirely in the robot/arena frame and is not affected.
The step-1 mag seed uses `mag_declination_static` which was calibrated against GT, so the
+π offset is already absorbed into that constant.

`tof_diagnostic.m` uses `gt_yaw + π` explicitly since it feeds GT heading directly into the
ToF geometry rather than through the mag seed formula.

---

## Arena Boundary Clamp

```matlab
X_u(1) = max(-Lx + 0.05, min(Lx - 0.05, X_u(1)));
X_u(2) = max(-Ly + 0.05, min(Ly - 0.05, X_u(2)));
```

Prevents position from escaping the arena. Without this, a heading error eventually makes
`h_pred ≤ 0` (ray parallel to or away from all walls), skipping all ToF updates and allowing
unbounded open-loop drift.
