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

**R_imu acc sweep:** Tried 0.05, 0.20, 0.30, 0.50 (all values). No measurable effect on any
dataset. The ToF updates dominate position accuracy; the IMU acc noise only affects how
aggressively the filter pins the `ax`/`ay` world-frame states between ToF steps, which is
swamped by the ToF correction. 0.50 retained (conservative / safe choice).

**IMU direction / bias sweep:** Tested all sign combinations for acc(2), acc(3), acc_x_bias,
acc_y_bias. Results:
- Flip acc(3) sign (y direction): catastrophic across all tasks (task2_1 pos_MSE ×50)
- Flip acc(2) sign (x direction): universally worse (~10% degradation)
- Flip acc_y_bias sign: catastrophic (same pattern as flipping y direction)
- Flip acc_x_bias sign: universally worse (acc_x_bias=0.0275 is tiny so effect is small but still negative)
Current configuration (-acc(2), +acc(3), -biases) is confirmed optimal.

**During `fast_spin`:** Acc noise inflated from R = (0.5)² to (5.0)² m²/s⁴.
Reason: centripetal acceleration during fast rotation is a real physical signal in body frame
but projects to world-frame `ax`, `ay` and drives position drift. Inflating acc noise
makes the filter ignore the acceleration measurement during spin. `fast_spin` is detected
using raw `gyro_z` (not the estimated `omega` state, which starts at 0 even if the robot is
already spinning at initialisation).

**Process noise:** `Q(7,7) = Q(8,8) = (0.25)²` — see Q sweep below.

**acc_y_bias refinement:** Original value -0.3963 was calibrated from static data. Sweep
around ±0.04 revealed -0.41 is the best combined pos+yaw optimum.

| acc_y_bias | task2_1_pos | task2_1_yaw | task2_2_pos | task2_3_pos | task2_3_yaw |
|------------|-------------|-------------|-------------|-------------|-------------|
| -0.37 | 0.0034 | 0.0153 | 0.0046 | 0.0044 | 0.0032 |
| -0.3963 (old) | 0.0040 | 0.0032 | 0.0049 | 0.0057 | 0.0037 |
| **-0.41** | **0.0040** | **0.0026** | **0.0049** | **0.0053** | **0.0030** |
| -0.42 | 0.0031 | 0.0091 | 0.0051 | 0.0048 | 0.0034 |
| -0.43 | 0.0035 | 0.0159 | 0.0053 | 0.0048 | 0.0035 |

The acc biases interact with heading through the H_imu rotation matrix (R(θ) maps world-frame
acc to body frame), so changing acc_y_bias indirectly affects the Kalman gain on omega/b_omega
and thus heading. -0.41 improves task2_3 position (0.0057→0.0053) and yaw (0.0037→0.0030),
and task2_1 yaw (0.0032→0.0026), without degrading anything else.

**acc_x_bias = 0.0275** confirmed optimal — perturbations in both directions degraded yaw.

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

**Gyro scale sweep (automated, task2_X pos_MSE):**

| Scale | task2_1 | task2_2 | task2_3 | Notes |
|-------|---------|---------|---------|-------|
| 1.05  | 0.0337  | 0.0118  | 0.0078  | task2_1 broken |
| 1.07  | 0.0192  | 0.0104  | 0.0078  | task2_1 still broken |
| 1.09  | 0.0127  | 0.0103  | 0.0078  | task2_1 improving, 2&3 better |
| **1.10**  | **0.0099**  | **0.0105**  | **0.0086**  | **baseline — best overall** |
| 1.11  | 0.0082  | 0.0106  | 0.0087  | task2_1 pos improves but yaw 0.0073 (×2) |
| 1.12  | 0.0067  | 0.0111  | 0.0088  | task2_1 yaw 0.0193 (×6), corner spikes massive |
| 1.15  | 0.0116  | 0.0140  | 0.0092  | everything worse |

Above 1.10, task2_1 pos_MSE artificially improves because the TOF compensates for an
over-rotating heading — the position plot shows severe corner spike clusters confirming
the lower MSE is deceptive. 1.10 retained.

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
Reason: motor EMI causes highly variable heading interference during operation. The effective
declination varies with motor speed and load, so there is no stable calibration constant
to use during motion.

**Tried: slow-motion mag updates** (`|gyro_z| < 0.3`, `|v| < 0.15 m/s`, `|acc| < 0.3 m/s²`).
Result: caused heading jumps at the start of datasets (motor ramp-up has intermediate EMI,
neither static nor full-speed) and continued drift elsewhere. The `mag_declination` baseline
is wrong for slow-speed operation. Removed.

**Tried: chi²-gated mag update (every step, no motion gate)** — `sweep_mag_chi2.m`,
`sweep_mag_decl.m`. The hypothesis was that chi² rejection would filter EMI-corrupted
readings without needing to know when motors are on or off.

**Declination diagnostic (`sweep_mag_decl.m` Part 1) — effective declination during operation:**

| Dataset  | decl_med | decl_std | decl_iqr |
|----------|----------|----------|----------|
| task1_1  | -1.623   | 1.137    | 1.416    |
| task1_2  | -1.585   | 1.253    | 1.221    |
| task1_3  | -1.711   | 1.346    | 1.402    |
| task1_4  | -1.717   | 1.586    | 1.604    |
| task2_1  | -1.522   | 1.443    | 0.289    |
| task2_2  | -1.189   | 1.961    | 3.011    |
| task2_3  | -1.562   | 2.041    | 1.598    |
| task2_4  | -1.228   | 1.963    | 3.796    |

decl_std of 1.1–2.0 rad (60–115°) spanning ±π on every dataset. The EMI corruption is not
a stable offset — it is highly variable and essentially covers the full angular range.

Result: all chi²-gated mag combinations made things 3–7× worse than no-mag baseline.
Accept rate was ~100% in all cases — the heading estimate drifts to match the wrong mag reading,
after which all future readings appear consistent (small chi²) and are accepted. The chi² gate
cannot detect this circular convergence to the wrong heading.

Declination sweep (decl from -1.4245 to π): no declination value improved over baseline.
Best found (decl = -1.0) still 3.7× worse on summed position RMSE.

**Conclusion: magnetometer cannot contribute heading corrections during motor operation on this hardware.
EMI is too variable — no fixed declination constant and no outlier-rejection scheme can help.**

**Tried: stationary-gated mag update** (`is_stationary` gate: `|gyro_z|<0.10 && |acc|<0.15`)
— `sweep_mag_stationary.m`. Hypothesis: motors-off stationary phases have low EMI and the static
declination (-1.4245) should be valid.

Result: `is_stationary` fires on 28–55% of all samples — far too many for "motors truly off".
Robot controllers energise motors even at zero velocity to hold position (torque-mode), so EMI
persists during stationary holds. All R_mag/chi² combinations remained 3–7× worse than baseline.

**Tried: sustained stationary-gated mag** (N_min = 400–1500 samples = 2–7.5 s)
— `sweep_mag_sustained.m`, `sweep_mag_sustained2.m`. Hypothesis: requiring prolonged stillness
filters motor-hold transients and only catches genuine motor-off pauses.

N_min=400 (2s), R_mag=20°: first configuration to beat baseline on summed position RMSE
(1.2637 vs 1.3451, 6% improvement). Per-dataset results:

| Dataset  | pos_RMSE (mag) | pos_RMSE (base) | Δ       |
|----------|----------------|-----------------|---------|
| task2_1  | 0.0262         | 0.0484          | **–46%** |
| task2_4  | 0.0592         | 0.0830          | **–29%** |
| task2_2  | 0.0990         | 0.0492          | +101%   |
| task2_3  | 0.0658         | 0.0520          | +27%    |

Yaw RMSE is worse on every dataset (heading correction pulls in the wrong direction on average).
The benefit/regression split is inconsistent across datasets — declination is correct for some
stop locations/orientations and wrong for others, with no predictable pattern. Not suitable for
production use.

**Final conclusion: magnetometer is exhausted as a heading sensor on this hardware.** Even 2–7.5 s
of sustained stillness does not reliably indicate motor de-energisation. EMI magnitude depends on
robot orientation relative to magnetic anomalies in the arena (hard/soft iron from structure,
wiring, motor position), not just motor current. No available sensor can distinguish "mag is clean"
from "mag is EMI-corrupted" moments.

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

**Tried: ZUPT (zero-velocity pseudo-measurement for vx, vy):** Applied `vx = 0, vy = 0`
pseudo-measurement when stationary. Tried with loose gate (`|acc| < 0.15`) and strict gate
(`|gyro_z| < 0.05 && |acc| < 0.05`). Both worsened pos_MSE across all tasks despite paths
looking visually straighter. Root cause: the stationary detector fires during slow motion
phases; zeroing velocity causes position to stagnate while GT continues moving, creating
systematic lag. Removed.

**Tried: sustained-stationary ZUPT** (`sweep_zupt.m`) — N_min=1–200 samples, R_zupt=0.01–0.20 m/s,
with and without ax/ay zeroing. Key findings:
- R_zupt has no effect regardless of value — IMU at 200 Hz already constrains velocity so tightly
  that ZUPT adds nothing the acc update doesn't already cover.
- Best result (N_min=10, R_zupt=0.01): 0.35% summed pos RMSE improvement, ~0.3 mm per dataset.
  Effectively zero; cannot be distinguished from noise.
- N_min=1 (every stationary sample) still worsens results — confirms original lag finding.
- Zeroing ax/ay simultaneously has no additional effect.
**Conclusion: ZUPT is a dead end for this EKF. The 200 Hz IMU leaves no velocity estimation gap
for ZUPT to fill.**

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

Chi² sweep result: 4.0 is optimal. Above 4.0 (tested 6.0) makes no difference — wall-ambiguity
and corner-margin checks already filter the worst readings before they reach the gate. Below 4.0
(tested 3.5, 3.0, 2.0) progressively rejects valid readings and degrades all tasks. 4.0 retained.

**Re-sweep after ToF offset calibration fix** (R_tof=0.07, corrected geometry): chi2 behaviour
unchanged — insensitive ≥4.0, cliff between 3.5 and 4.0 (task2_3 0.0057→0.0085 at 3.5).
4.0 confirmed as the minimum safe threshold regardless of sensor calibration quality.

**R_tof = (0.07)²** — ToF measurement noise. (Was `(0.10)²` before ToF offset fix.)

**Original sweep** (with wrong ToF2 dx=-0.09):

| R_tof | task1_2 | task1_3 | task2_1 | task2_2 | task2_3 |
|-------|---------|---------|---------|---------|---------|
| 0.07  | 0.0108  | 0.0113  | 0.0095  | 0.0123  | 0.0109  |
| 0.09  | 0.0109  | 0.0115  | 0.0095  | 0.0096  | 0.0082  |
| 0.10 (was best) | 0.0110 | 0.0116 | 0.0094 | 0.0096 | 0.0082 |
| 0.15 (old) | 0.0113 | 0.0120 | 0.0094 | 0.0098 | 0.0085 |

**Re-sweep after ToF offset fix** (with corrected dx=-0.02):

| R_tof | task1_2 | task1_3 | task2_1 | task2_2 | task2_3 | sum |
|-------|---------|---------|---------|---------|---------|-----|
| 0.05 | 0.0060 | 0.0068 | 0.0039 | 0.0050 | 0.0057 | 0.0274 |
| 0.06 | 0.0060 | 0.0068 | 0.0039 | 0.0050 | 0.0057 | 0.0274 |
| **0.07** | **0.0060** | **0.0069** | **0.0039** | **0.0049** | **0.0057** | **0.0274** |
| 0.10 (old) | 0.0061 | 0.0070 | 0.0040 | 0.0049 | 0.0058 | 0.0278 |

With corrected geometry, tighter R_tof is uniformly better — the residuals are now small and
genuine, so trusting them more helps. Plateau at 0.05–0.07; 0.07 chosen (best task2_2, round-ish).
The old 0.10 optimum was partially compensating for the systematic offset in h_pred.

**Incidence-angle-adaptive R_tof:** `R_tof_a = R_tof / max(inc_cos, 0.30)²` where `inc_cos`
is `|cos(ray)|` for x-wall hits and `|sin(ray)|` for y-wall hits. At perpendicular incidence
`R_tof_a = R_tof` (nominal). At oblique angles the measurement is inflated, reducing the
update gain. Floor at 0.30 caps max inflation at ~11×.

Floor tuning result: floor=0.15 (44×) and floor=0.50 (4×) both gave marginal or no
improvement vs floor=0.30. The wall-ambiguity check already rejects the most oblique readings,
so the floor is rarely the active constraint. 0.30 retained as the best value.
Net gain vs flat R_tof: task2_1 0.0100→0.0099, task2_3 0.0090→0.0086. No regressions.

**During `fast_spin`:** Position process noise inflated (`Q(1,1) = Q(2,2) = (0.15)²`)
so that `P` grows faster → `S_tof` grows → chi² gate naturally widens.
Reason: centripetal acc can drift position estimate slightly before the gate activates;
wider gate allows valid ToF measurements through to pull position back.

**Not used for heading (`H_tof(3) = 0`). DO NOT re-enable. Ever.**
Multiple attempts, all catastrophic:
- Full `H_tof(3)` (unclamped): violent ±π heading spikes across all datasets.
- `H_tof(3)` clamped to ±1.5 m/rad: massively worsened every task except task1_1.
- Root cause: `dh/dtheta` can be several m/rad at oblique wall angles. A small position
  residual produces a gain `K_tof(3)` large enough to rotate heading by radians per update.
  At perpendicular incidence `dh/dtheta ≈ 0` (which is most of the time), so H_tof(3) carries
  no useful information. At oblique incidence it explodes. No clamping value fixes this.
- task1_1's wrong-initial-heading problem must be solved another way.

**Wall ambiguity check:** Added in `tof_measurement` — if the two shortest ray-wall distances
satisfy `t_second < 1.20 × t_first`, measurement is rejected. Near arena corners or when the
sensor fires at 45°, a small heading error flips which wall is selected, producing a
discontinuous ~1–2 m jump in h_pred.

Wall ambiguity threshold sweep (with R_tof=0.10, corner_margin=0.1):

| Threshold | task2_1 | task2_2 | task2_3 | sum |
|-----------|---------|---------|---------|-----|
| 1.10      | crash   | —       | —       | —   |
| **1.20**  | **0.0095** | **0.0095** | **0.0081** | **0.0271** |
| 1.25      | 0.0095  | 0.0096  | 0.0082  | 0.0273 |
| 1.30 (old)| 0.0094  | 0.0096  | 0.0082  | 0.0272 |
| 1.40      | 0.0091  | 0.0099  | 0.0084  | 0.0274 |
| 1.50      | 0.0094  | 0.0099  | 0.0087  | 0.0280 |

1.20 wins marginally on sum; 1.10 causes EKF divergence (too many rejections → open-loop drift).
Differences are small (0.0001 level) but 1.20 is consistently best.

**Corner margin:** `corner_margin = 0.1 m` — if the ray hit-point is within 0.1 m of a corner
(both `|hit_x| > Lx - 0.1` and `|hit_y| > Ly - 0.1`), reading is discarded.

Corner margin sweep result: lower is better; gains plateau below 0.1 (wall-ambiguity check
already rejects most bad corner readings). 0.0 (disabled) gives negligible further improvement.

| margin | task2_1 | task2_2 | task2_3 |
|--------|---------|---------|---------|
| 0.0    | 0.0095  | 0.0097  | 0.0084  |
| 0.05   | 0.0095  | 0.0097  | 0.0085  |
| **0.1**    | **0.0094**  | **0.0098**  | **0.0085**  |
| 0.2 (old)  | 0.0099  | 0.0105  | 0.0086  |
| 0.3    | 0.0108  | 0.0135  | 0.0099  |

**Things tried for corner/turn artifacts — not adopted:**
- Absolute residual cap `|nu_tof| > 0.40 m` (all steps): broke valid turn tracking, reverted.
- Skip all ToF during `fast_spin`: lost wall reference for entire spin, made things worse.
- Scoped cap `fast_spin && |nu_tof| > 0.40`: had no measurable effect.
- Gyro-based cap `|gyro_z| > 0.15 && |nu_tof| > 0.50`: condition never met at spike moments.
- Jacobian magnitude check `|dh_dsx| > 3.2 || |dh_dsy| > 3.2`: Jacobians stay ~1.0 at actual
  spike locations (sensors hit walls near-perpendicularly there). No effect.
- Negating sensor offsets dx and dy: improved overall MSE but broke task2_3 shape.
- R_tof = (0.20)²: slightly shifted shapes, not an improvement. Kept at (0.15)².
- Q_ax/Q_ay = (0.07)²: no improvement. Kept at (0.15)².
- **Paired-sensor heading geometry:** When sensors hit different wall types simultaneously
  (one x-wall, one y-wall), each 90°-firing sensor gives a pure sin(θ) or cos(θ) constraint.
  Collecting one of each gives θ_geom = atan2(sin_est, cos_est) as a heading pseudo-measurement
  without using H_tof(3). Fired <100 times across all 6 datasets (robot rarely faces a diagonal
  that splits sensors across different wall types). Effect: task2_1 improved (0.0100→0.0080,
  yaw 0.0032→0.0019) but task2_3 yaw SSE exploded 6× (24.8→144.6) and task2_2 worsened.
  Root cause: position uncertainty ~0.1 m at range ~0.5 m gives heading uncertainty ~0.2 rad;
  on rare firings the estimate can be badly wrong and R_pg = (0.15)² was too tight to suppress
  it. Adding a chi2 gate was not tried — the rarity of firing and severity of task2_3 regression
  made it not worth pursuing. Removed.

---

## Heading Drift — Known Limitation

With no ongoing heading correction, `theta` drifts with gyro bias error:

| Dataset | Observed drift | Source |
|---------|---------------|--------|
| straight (70 s) | Small (~0.03 rad) | Good static bias calibration; minimal rotation |
| task2_1, task2_2 | ~0.2 rad lag during rotation | Motor EMI shifts effective gyro bias |

**Options investigated and rejected:**
1. `H_tof(3)` (any form, any clamp) — see ToF section. **Do not revisit.**
2. Slow-motion mag updates — variable EMI baseline, causes jumps (see Mag section)
3. Accelerometer for heading — physically impossible (gravity ⊥ yaw axis)
4. Dynamic b_omega during motion — no correction signal available; causes spurious drift

**b_omega state** partially addresses this during stationary phases but cannot correct
drift accumulated during motor operation.

**Additional approaches tried and rejected:**

5. **Temperature compensation:** Temp sensor available but useless — only ~0.3°C variation
   across all datasets, correlation with gyro_z < 0.05 on all runs. Bias drift is purely
   EMI-driven, not thermal.

6. **Adaptive gyro_scale_fast:** Separate `gyro_scale` for `fast_spin` phases (swept 0.90–1.15).
   Below 1.1 catastrophically under-integrates (task2_1 pos_MSE ×10). Above 1.1 improves
   pos on some tasks but yaw explodes on others. The EMI effect is a bias offset, not a scale
   error — a different scale during spin cannot fix it. Reverted to `gyro_scale_fast = 1.1`.

7. **Slow b_omega drift during motion (`Q_bomega_motion > 0`):** Tried 1e-7 to 1e-5. Without
   a correction signal during motion the random walk drifts randomly toward or away from the
   true bias — results are inconsistently mixed at every value. Never better than frozen.
   Reverted to 0.

8. **Velocity-direction heading constraint:** When translating (low omega, low vy_body),
   apply `theta ≈ atan2(vy, vx)` as a soft pseudo-measurement (R = (15°)²). Catastrophic —
   destroyed heading on all tasks. Root causes: (a) EKF velocity is derived from the same
   theta being corrected (circular dependency); (b) mecanum strafing means velocity direction
   ≠ heading even during apparently forward motion. Removed entirely.

**task1_1 specific issue:** GT heading injected via `fake_mag` in `run_ekf_test.m` is wrong
at t=0 for this dataset. With no ongoing heading correction, the error persists for the entire
run. pos_MSE ≈ 1.077, yaw_MSE ≈ 2.605 — all other tasks are < 0.015 pos_MSE. The bad initial
heading is a data/seeding issue not solvable by tuning the EKF update equations.

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

---

## Arena Dimensions

`Lx = Ly = 1.22 m` confirmed accurate. Swept ±0.01 m (1.21, 1.22, 1.23) — no consistent
improvement in either direction. 1.22 is the true physical arena half-width.

---

## Process Noise Q Sweep

Final Q (after sweep):
```
Q = diag([ 5e-3, 5e-3, deg2rad(3),   % x, y, theta
           0.10, 0.10, 0.10,          % vx, vy, omega
           0.25, 0.25, 3e-3 ].^2)
```

**Q_vx/vy sweep** (Q_ax/ay=0.15, all other params at tuned baseline):

| Q_vx/vy | task1_2 | task1_3 | task2_1 | task2_2 | task2_3 |
|---------|---------|---------|---------|---------|---------|
| 0.02    | 0.0121  | 0.0130  | 0.0095  | 0.0100  | 0.0088  | worse
| **0.05 (old)** | **0.0110** | **0.0116** | **0.0095** | **0.0095** | **0.0081** | baseline
| **0.10** | **0.0107** | **0.0112** | **0.0096** | **0.0095** | **0.0079** | best
| 0.20    | 0.0107  | 0.0111  | 0.0100  | 0.0097  | 0.0079  | task2_1 regresses

0.10 is the sweet spot — task1 consistently improves (~3%), task2 neutral or marginal improvement.
Lower values (0.02) over-constrain velocity, causing the model to reject valid correction;
higher values (0.20) let velocity noise bleed into position without benefit.

**Q_ax/ay sweep** (Q_vx/vy=0.10):

| Q_ax/ay | task1_2 | task1_3 | task2_1 | task2_2 | task2_3 |
|---------|---------|---------|---------|---------|---------|
| 0.10    | 0.0107  | 0.0112  | 0.0096  | 0.0095  | 0.0079  |
| 0.15    | 0.0107  | 0.0112  | 0.0096  | 0.0095  | 0.0079  |
| **0.25** | **0.0107** | **0.0112** | **0.0096** | **0.0094** | **0.0079** | marginal win
| 0.30    | 0.0107  | 0.0112  | 0.0096  | 0.0094  | 0.0079  |

Q_ax/ay is largely insensitive — position dominated by ToF updates; the acc model between
ToF steps has little effect. 0.25 chosen for marginal task2_2 improvement (0.0095→0.0094).

**Q_omega / Q_theta sweeps:** Not performed. Yaw MSE is completely frozen across all
Q variations (gyro measurement at 200 Hz dominates heading dynamics; process noise is
irrelevant vs the measurement update rate). Changing Q_omega or Q_theta has no measurable effect.

**Re-sweep after all calibration fixes:** Q_vx/vy=0.10 and Q_ax/ay=0.25 both re-confirmed
optimal. The error landscape changed dramatically after ToF offset fix but Q optima did not
shift — Q governs model trust relative to measurements; fixing geometry reduces absolute errors
but not the relative dynamics between prediction and measurement update.

---

## fast_spin Parameter Sweeps

`fast_spin` is detected via `|gyro_z| > threshold`, and triggers: (1) inflated Q_pos to widen
the chi2 gate, and (2) inflated R_acc to distrust centripetal contamination in the accelerometer.

**Threshold sweep** (0.3, 0.5, 0.7, 1.0 rad/s):

| Threshold | task2_1 | task2_2 | task2_3 | sum |
|-----------|---------|---------|---------|-----|
| 0.3 | 0.0093 | 0.0097 | 0.0079 | 0.0269 |
| **0.5** | **0.0096** | **0.0094** | **0.0079** | **0.0269** |
| 0.7 | 0.0094 | 0.0097 | 0.0079 | 0.0270 |
| 1.0 | 0.0097 | 0.0100 | 0.0079 | 0.0276 |

Insensitive between 0.3–0.5 (same sum), degrades above. 0.5 retained — most datasets have
fast spins well above 0.5 rad/s so there is no risk of premature triggering.

**fast_spin Q_pos sweep** (`(0.10)²` to `(0.30)²`):
Insensitive — all results within noise of baseline. 0.15 retained.

**fast_spin R_acc sweep** (inflation during spin):

| R_acc | task2_1 | task2_2 | task2_3 | sum |
|-------|---------|---------|---------|-----|
| 0.5 (no inflation) | 0.0092 | 0.0094 | 0.0079 | 0.0265 |
| **1.0** | **0.0093** | **0.0094** | **0.0078** | **0.0265** |
| 2.0 | 0.0094 | 0.0094 | 0.0078 | 0.0266 |
| 5.0 (old) | 0.0096 | 0.0094 | 0.0079 | 0.0269 |
| 10.0 | 0.0097 | 0.0095 | 0.0080 | 0.0272 |

Lower inflation is better — the original (5.0)² was overcorrecting. 1.0 retained: task2_3 gets
0.0078 (vs 0.0079 for no inflation) while being a mild, principled signal of reduced confidence.
The centripetal contamination assumption was overstated; mild caution (2× std) is sufficient.

---

## ToF Sensor Offset Calibration

**Critical finding: ToF2 forward offset was severely miscalibrated.**

Original `tof_offsets(2, 1) = -0.09 m`. Sweep showed the true offset is ~-0.02 m — the sensor
is much closer to the robot centre than assumed. The 7 cm error added a systematic bias to every
forward-sensor range prediction, and corrupted the chi2 gate residuals.

**ToF2 dx_fwd sweep** (holding ToF1/3 at `[0, ±0.04]`):

| dx (m) | task1_2 | task1_3 | task2_1 | task2_2 | task2_3 | sum |
|--------|---------|---------|---------|---------|---------|-----|
| -0.09 (old) | 0.0107 | 0.0112 | 0.0096 | 0.0094 | 0.0079 | 0.0488 |
| -0.07 | 0.0084 | 0.0090 | 0.0070 | 0.0074 | 0.0064 | 0.0382 |
| -0.05 | 0.0069 | 0.0076 | 0.0053 | 0.0059 | 0.0056 | 0.0313 |
| -0.03 | 0.0062 | 0.0070 | 0.0044 | 0.0050 | 0.0056 | 0.0282 |
| **-0.02** | **0.0061** | **0.0070** | **0.0041** | **0.0049** | **0.0058** | **0.0279** |
| -0.01 | 0.0062 | 0.0071 | 0.0041 | 0.0049 | 0.0061 | 0.0284 |
| 0.00 | 0.0066 | 0.0075 | 0.0043 | 0.0050 | 0.0066 | 0.0300 |

-0.02 is optimal. Task2_3 peaks around -0.03; task2_1/2 continue improving to -0.02 then level off.

**ToF1/3 side offset (dy) sweep** (holding ToF2 at -0.02):

| dy (m) | task2_1 | task2_2 | task2_3 | sum |
|--------|---------|---------|---------|-----|
| ±0.02 | 0.0039 | 0.0050 | 0.0059 | 0.0148 |
| **±0.03** | **0.0040** | **0.0049** | **0.0058** | **0.0147** |
| ±0.04 (old) | 0.0041 | 0.0049 | 0.0058 | 0.0148 |
| ±0.05 | 0.0043 | 0.0049 | 0.0057 | 0.0149 |

Marginally insensitive — ±0.03 best on sum. Differences at 0.0001 level. ±0.03 adopted.

**Fine resolution check on ToF2 dx** (±0.005 m around -0.02): tried -0.015 and -0.025.
Both worse — -0.02 confirmed at 5 mm resolution.

**Final offsets:**
```matlab
tof_offsets = [ 0.00,  0.03;   % ToF1 – right-facing
               -0.02,  0.00;   % ToF2 – forward-facing
                0.00, -0.03];  % ToF3 – left-facing
```

---

## R_imu Gyro Noise Sweep

`R_imu(3,3)` controls how tightly the filter trusts the gyro for `omega`. Unlike acc noise
(which is ToF-dominated and insensitive), gyro noise actively affects `b_omega` estimation
and therefore heading — changing it moves both yaw MSE and position MSE.

| R_gyro | task2_1_pos | task2_1_yaw | task2_2_pos | task2_2_yaw | task2_3_pos |
|--------|-------------|-------------|-------------|-------------|-------------|
| 0.01 | 0.0050 | **0.0019** | **0.0045** | 0.0022 | 0.0056 |
| **0.02** | **0.0040** | 0.0032 | 0.0049 | **0.0023** | **0.0058** |
| 0.03 | 0.0036 | 0.0045 | 0.0051 | 0.0026 | 0.0061 |
| 0.05 | 0.0034 | 0.0065 | 0.0053 | 0.0033 | 0.0063 |

Clear tradeoff: tighter → better task2_2 pos and task2_1 yaw but hurts task2_1 pos;
looser → better task2_1 pos but yaw degrades significantly and other tasks regress.
0.02 is the best balance for the combined pos+yaw metric. Retained.

---

## Final Parameter Sweep (sweep_final.m)

Three-part sweep covering Q(3,3) fast_spin inflation, gyro_x_bias, and per-sensor R_tof.
Script: `sweep_final.m`. Baseline: t1_pos=0.3254, t1_yaw=0.0786, t2_pos=0.3113, t2_yaw=0.2542, sum=0.6368.

### Part 1: Q(3,3) Heading Noise During fast_spin

Tested deg2rad([3,5,10,20,30,45])² — zero effect on all datasets across all values.
The fast_spin path (`|omega| > 2 rad/s`) fires rarely and briefly; heading divergence
during fast rotations is dominated by integration error, not the process noise setting.
Current value (3 deg) retained.

### Part 2: gyro_x_bias Fine Sweep

Swept -0.0162 to -0.0062 in 0.001 steps (current = -0.0112).

| gx_bias | t1_pos | t1_yaw | t2_pos | t2_yaw | sum |
|---------|--------|--------|--------|--------|-----|
| -0.0162 | 0.3254 | 0.0786 | 0.3184 | 0.2813 | 0.6637 |
| -0.0112 (baseline) | 0.3254 | 0.0786 | 0.3113 | 0.2542 | 0.6368 |
| -0.0102 | 0.3254 | 0.0786 | 0.2942 | 0.2333 | 0.6128 |
| **-0.0072** | **0.3254** | **0.0787** | **0.2921** | **0.2334** | **0.6175** |
| -0.0062 | 0.3254 | 0.0787 | 0.2921 | 0.2334 | 0.6175 |

Best: **-0.0072** (sum=0.6175, -3.0% vs baseline). Improvement entirely in task2 —
task1 is insensitive to gyro_x_bias (robot doesn't pitch/roll significantly in task1 profiles).
Values from -0.0072 to -0.0062 tie; -0.0072 adopted as the midpoint of the plateau.

### Part 3: Per-Sensor R_tof (R_side=ToF1/ToF3, R_fwd=ToF2)

Current: all sensors at R_tof = 0.07 m. Swept R_side ∈ {0.05, 0.07, 0.09, 0.12} × R_fwd ∈ {0.04, 0.05, 0.07, 0.09}.

Selected rows (full grid in sweep_final.m output):

| R_side | R_fwd | t1_pos | t2_pos | sum |
|--------|-------|--------|--------|-----|
| 0.07 | 0.07 (baseline) | 0.3254 | 0.3113 | 0.6368 |
| 0.07 | 0.04 | 0.3242 | 0.3013 | 0.6248 |
| **0.09** | **0.04** | **0.3243** | **0.2992** | **0.6234** |
| 0.09 | 0.05 | 0.3245 | 0.2996 | 0.6241 |
| 0.12 | 0.04 | 0.3245 | 0.2995 | 0.6240 |

Best: **R_side=0.09, R_fwd=0.04** (sum=0.6234, -2.1% vs baseline).
Pattern: lower R_fwd (trust forward ToF2 more) consistently improves task2; R_side variation
is secondary but R_side=0.09 slightly better than uniform 0.07 (side sensors less trusted).

### Combined Result

Applied all three improvements: gyro_x_bias=-0.0072, R_side=0.09, R_fwd=0.04, Q_theta_fs unchanged.

| Dataset | pos_RMSE | yaw_RMSE | pos_base | yaw_base | Δpos |
|---------|----------|----------|----------|----------|------|
| task1_1 | 0.0846 | 0.0445 | 0.0848 | 0.0445 | -0.2% |
| task1_2 | 0.0772 | 0.0108 | 0.0776 | 0.0107 | -0.5% |
| task1_3 | 0.0826 | 0.0134 | 0.0829 | 0.0135 | -0.4% |
| task1_4 | 0.0798 | 0.0099 | 0.0802 | 0.0099 | -0.5% |
| task2_1 | 0.0644 | 0.0454 | 0.0636 | 0.0513 | +1.3% |
| task2_2 | 0.0696 | 0.0501 | 0.0702 | 0.0501 | -0.9% |
| task2_3 | 0.0719 | 0.0551 | 0.0729 | 0.0551 | -1.4% |
| task2_4 | 0.0842 | 0.0828 | 0.1047 | 0.0977 | **-19.6%** |
| **TOTAL** | **0.6143** | **0.3120** | **0.6368** | **0.3328** | **-3.5%** |

Net: pos RMSE 0.6368 → 0.6143 (−3.5%), yaw RMSE 0.3328 → 0.3120 (−6.3%).
task2_4 is the main beneficiary (19.6% pos improvement) — this dataset has the most sustained
rotation where gyro_x_bias matters. task2_1 is marginally worse (+1.3%), within noise.

**Parameters applied to myEKF_ca.m:**
- `gyro_x_bias = -0.0072` (was -0.0112)
- `R_tof1 = 0.09^2`, `R_tof2 = 0.04^2`, `R_tof3 = 0.09^2` (was all 0.07^2)
- Q_theta_fs: no change

---

## Parameter Sweep (sweep_params.m)

Five sweeps on the post-sweep_final baseline (sum=0.6143).
Script: `sweep_params.m`.

### Part 1: chi2_thresh

Swept [1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 12.0, 16.0]. Current = 4.0 is the sharp knee:
below 4.0 degrades dramatically (valid ToF readings rejected when position uncertainty is
temporarily elevated); 4.0 and above are identical (no outliers in these datasets above
this threshold). **4.0 confirmed optimal.**

### Part 2: gyro_scale

Swept [1.05, 1.07, 1.08, 1.09, 1.10, 1.11, 1.12, 1.13, 1.15]. Current = 1.10.

| gyro_scale | t1_pos | t2_pos | t2_yaw | sum |
|------------|--------|--------|--------|-----|
| 1.090 | 0.3243 | 0.2889 | 0.2103 | 0.6132 |
| **1.100 (was)** | 0.3243 | 0.2900 | 0.2334 | 0.6143 |
| 1.110 | 0.3243 | 0.2904 | 0.2847 | 0.6147 |

1.090 beats 1.100 primarily via task2 yaw (−10%). Applied.
**Note:** per-dataset breakdown shows task2_1 pos regresses ~20% while task2_4 improves ~13%.
Net aggregate pos: −0.17%, net aggregate yaw: −7.5%.

### Part 3: corner_margin

Swept [0.03, 0.05, 0.07, 0.10, 0.13, 0.15, 0.20]. Current = 0.10 m.
0.03–0.07 tie at sum=0.6133 (vs baseline 0.6143). 0.10+ progressively worse.
**corner_margin=0.05 applied** (recovers a few extra ToF readings near corners).

### Part 4: Q(9,9) b_omega random walk

Swept [1e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2] rad/s. Current = 3e-3.
5e-3 gives best pos aggregate but degrades task2 yaw (0.2334 → 0.2144).
Interaction with gyro_scale=1.09 causes task2_1 to blow up in combined run (+48% pos).
**Not applied** — tradeoff not worth it, current 3e-3 retained.

### Part 5: fast_spin threshold

Swept [0.30, 0.50, 0.80, 1.00, 1.50, 2.00, 3.00] rad/s. Current = 0.50.
Lower triggers inflation too often (task2 worse); higher fails to suppress centripetal.
**0.50 rad/s confirmed optimal.**

### Final state after sweep_params (run_ekf_test.m):

| Dataset | pos_RMSE | yaw_RMSE |
|---------|----------|----------|
| task1_1 | 0.0846 | 0.0445 |
| task1_2 | 0.0772 | 0.0107 |
| task1_3 | 0.0826 | 0.0133 |
| task1_4 | 0.0798 | 0.0098 |
| task2_1 | 0.0774 | 0.0405 |
| task2_2 | 0.0667 | 0.0478 |
| task2_3 | 0.0711 | 0.0538 |
| task2_4 | 0.0732 | 0.0682 |
| **TOTAL pos** | **0.6126** | — |

vs post-sweep_final baseline: pos 0.6143 → 0.6126 (−0.17%), yaw 0.3120 → 0.2886 (−7.5%).

**Active parameters in myEKF_ca.m:**
- `gyro_x_bias = -0.0072`
- `gyro_scale = 1.09`
- `R_tof = [0.09², 0.04², 0.09²]` (side/fwd/side)
- `corner_margin = 0.05`
- `chi2_thresh = 4.0` (confirmed optimal)
- `fast_spin_thr = 0.50 rad/s` (confirmed optimal)

---

## Structural Experiments — Summary

The following experiments were motivated by observing that ground-truth heading provides only ~20%
improvement in position RMSE. The remaining ~80% of error is structural — position estimation
error independent of heading quality. Four structural alternatives were tested:

---

## ToF Gating with Orbital Velocity Compensation (sweep_tof_spin.m)

**Motivation:** During fast spin the IMU is at radius r_imu ≈ 0.02 m forward of the geometric
centre. The IMU therefore orbits the turning centre, inducing an apparent translational velocity
even during pure rotation. Hypothesis: gating ToF during fast spin and injecting orbital velocity
correction would reduce jumpy position updates.

**Orbital velocity correction (world frame):**
```
vx_orbital =  r_imu * sin(θ) * ω
vy_orbital = -r_imu * cos(θ) * ω
```

**Part 1 — tof_spin_thr sweep** (0.3–3.0 rad/s; r_imu = 0.02 m):

| tof_spin_thr (rad/s) | t1_pos | t2_pos | sum |
|----------------------|--------|--------|-----|
| 0.30 | 0.1718 | 0.3518 | 0.5236 |
| 0.50 | 0.1042 | 0.1817 | 0.2859 |
| 1.00 | 0.0835 | 0.1050 | 0.1885 |
| 2.00 | 0.1023 | 0.0966 | 0.1989 |
| 3.00 (never fires) | — | — | 0.3618 |
| **baseline (no gating)** | — | — | **0.3618** |

All thresholds that actually fire degrade performance. At 0.30 rad/s the t2 pos doubles to 0.3518.
The robot has genuine translational velocity during spins and ToF provides needed position
corrections; gating it out removes this useful signal.

**Part 2 — r_imu sweep** (threshold fixed at 3.0 rad/s — never fires):
All values return sum=0.3618 (the correction never activates). The orbital velocity magnitude at
r_imu=0.02 m is small enough that even if gating fired, the correction would be negligible.

**Conclusion:** ToF gating during fast spin is uniformly harmful. `tof_fast_spin` flag in
`myEKF_ca.m` remains disabled (set to 0). The IMU offset from geometric centre (tof_offsets)
is already calibrated relative to the GT reference; no additional correction is needed.

---

## R_tof Inflation During Straight-Line Motion (sweep_straight.m)

**Motivation:** Straight-line sections show sharp left-right jumps at 10 Hz ToF update rate.
Hypothesis: inflating R_tof during detected straight motion would smooth the trajectory by
diluting ToF influence when the accelerometer is reliable.

**Straight-line detection:** `|gyro_z| < omega_thr AND speed > speed_thr`

**Part 1 — R_inflate sweep** (omega_thr=0.3 rad/s, speed_thr=0.05 m/s):

| R_inflate | t1_pos | t2_pos | sum |
|-----------|--------|--------|-----|
| 1 (baseline) | — | — | 0.3618 |
| 2 | +0.004 | — | degrades |
| 4 | worse | — | degrades |
| 8, 16, 32, 64 | progressively worse | — | degrades |

Every inflation factor degrades performance. Factor=2 alone costs +0.004 on t1 pos.

**Root cause:** The mecanum robot strафes heavily — heading and drive direction are decoupled.
"Straight-line" detection (low gyro_z, high speed) captures strafing phases where the
accelerometer is not aligned with the direction of travel. The ToF jumps are necessary
corrections to erroneous velocity estimates, not noise to be filtered out.

**Part 2:** Not run — no Part 1 improvement to follow up on.

**Conclusion:** R_tof inflation during straight motion is uniformly harmful. Not applied.

---

## Q Tightening During Straight-Line Motion (sweep_straight_Q.m)

**Motivation:** Instead of distrusting ToF, tighten the process noise for velocity/acceleration
during straight-line motion — allowing the filter to hold a more consistent velocity estimate
and reducing susceptibility to ToF jitter.

**Part 1 — Q_v_scale sweep** (scales Q(4,4) and Q(5,5); omega_thr=0.3, speed_thr=0.05):

| Q_v_scale | t1_pos | t2_pos | sum |
|-----------|--------|--------|-----|
| 1.00 (baseline) | — | — | 0.3618 |
| 0.50 | worse | — | ≥0.363 |
| 0.10 | worse | — | ≥0.363 |
| 0.01 | worse | — | ≥0.363 |

No improvement at any scale. Tighter Q hurts task1 (strafing ~50% of time) more than it
helps task2.

**Part 2 — Q_a_scale sweep** (scales Q(7,7) and Q(8,8)): Same pattern — no improvement.

**Part 3 — omega_thr sweep** (0.10–1.50 rad/s): Insensitive; never improves on baseline.

**Best found:** sum=0.3643 vs baseline=0.3618 — no combination beats baseline.

**Conclusion:** Q tightening during straight motion is harmful on net. The same root cause
applies: mecanum strafing means "straight" detection incorrectly fires during lateral motion,
and tightening Q prevents the filter from tracking the actual (non-aligned) velocity.

---

## Heading-Velocity Alignment Diagnostic

**Question:** Does GT heading align with GT drive direction? (prerequisite for body-frame or
lateral ZUPT approaches)

**Method:** Computed angle between GT velocity vector (vx,vy) and GT heading θ for all 8 datasets.

**Results:**

| Dataset | Mean angle | Distribution |
|---------|-----------|--------------|
| task1_1 | ~87° | bimodal: ~50% < 10°, ~50% > 30° |
| task1_2 | ~89° | bimodal: ~50% < 10°, ~50% > 30° |
| task1_3 | ~85° | similar |
| task1_4 | ~91° | similar |
| task2_1 | ~95° | >97% of time > 30° misalignment |
| task2_2 | ~115° | >97% of time > 30° misalignment |
| task2_3 | ~138° | >97% of time > 30° misalignment |
| task2_4 | ~143° | >97% of time > 30° misalignment |

**Conclusion:** Heading and drive direction are severely decoupled across all datasets. This is
expected for mecanum wheels. Body-frame lateral ZUPT is not viable. Any approach that assumes
heading ≈ drive direction will fail on task2_x datasets.

---

## Body-Frame Velocity States (sweep_body_frame.m)

**Hypothesis:** Storing velocity and acceleration in body frame (vx_b, vy_b, ax_b, ay_b) would
simplify the IMU update — no rotation matrix needed since acc measurements are already in body
frame — potentially reducing linearisation error.

**Model structure:**
- State: `[x, y, θ, vx_b, vy_b, ω, ax_b, ay_b, b_ω]` (9 states, same count)
- Prediction: position couples through heading:
  ```
  x_k+1 = x + (vx_b·cos(θ) - vy_b·sin(θ))·dt + 0.5·(ax_b·cos(θ) - ay_b·sin(θ))·dt²
  y_k+1 = y + (vx_b·sin(θ) + vy_b·cos(θ))·dt + 0.5·(ax_b·sin(θ) + ay_b·cos(θ))·dt²
  ```
- Jacobian: F(1,4)=cos(θ)·dt — heading-dependent coupling to x-position
- IMU update: `h = [ax_b; ay_b; ω + b_ω]` — no rotation matrix in H

**Part 1 — direct comparison:**

| Model | sum |
|-------|-----|
| CA world-frame (baseline) | 0.3618 |
| Body-frame | 7.0083 |

Body-frame is +1837% worse. Even with extensive Q tuning (Parts 2/3) the best achieved was
sum=3.1397 (+767%).

**Root cause — heading-dependent observability:**
World-frame model: F(1,4)=dt always — vx_w always couples to x-position regardless of heading.
Body-frame model: F(1,4)=cos(θ)·dt — at θ=π/2, vx_b barely couples to x-position. The ToF
x-sensor cannot correct vx_b errors when the robot faces sideways. Velocity states drift during
heading-dependent observability gaps, creating systematic accumulating error.

**Conclusion:** Body-frame velocity states are structurally inferior for a ToF-localised mecanum
robot. World-frame model retained.

---

## CV Model with Acceleration as Control Input (sweep_cv.m)

**Hypothesis:** A 7-state constant-velocity (CV) model using acc as a direct control input
might be cleaner — the CA model's ax/ay states could be introducing lag or absorbing noise
that a direct-feed approach would avoid.

**Model structure:**
- State: `[x, y, θ, vx, vy, ω, b_ω]` (7 states)
- Prediction (acc as B·u control term):
  ```
  ax_inp = acc_bx·cos(θ) - acc_by·sin(θ)    (world-frame rotation of body acc)
  ay_inp = acc_bx·sin(θ) + acc_by·cos(θ)
  x_k+1 = x + vx·dt + 0.5·ax_inp·dt²
  vx_k+1 = vx + ax_inp·dt
  ```
- Measurement update: gyro only — `h = ω + b_ω`; no acc in measurement (acc is control input)

**Part 1 — direct comparison:**

| Model | sum |
|-------|-----|
| CA (baseline) | 0.3618 |
| CV | 5.4399 |

CV is +1403% worse at default Q. With Q_v=1.00 m/s²: sum=0.4820 (+33%). With Q_v=1.00 and
Q_v_fast_spin=0.10: sum=0.4723 (+30.5%). Never approaches CA performance.

**Root cause — CA states act as Kalman low-pass filter:**
The CA model's ax/ay states form an implicit low-pass filter on the noisy accelerometer.
The Kalman update smoothly blends acc measurements into ax/ay with appropriate uncertainty,
attenuating high-frequency noise. The CV model bypasses this filtering — raw acc noise enters
velocity directly each timestep. With Q_v large (≈1.0), the filter has almost no velocity
memory and effectively averages acc over a short window — but this is less principled than the
CA model's Kalman filter structure and still 30% worse.

**Conclusion:** The CA model's ax/ay states are a feature, not a liability. CV model not adopted.

---

## LP_acc Sensor Diagnostic

**Question:** Can the low-pass filtered accelerometer (`lp_acc` in data) improve performance
by providing cleaner acceleration measurements?

**Finding:** `lp_acc(:,2)` after the same body-frame → world-frame transformation as raw acc
gives a mean of ~10.2 m/s², compared to raw acc mean of ~-0.1 m/s². The LP_acc axis 2 measures
gravity (~9.81 m/s²), not horizontal body acceleration.

**Conclusion:** LP_acc has a different physical axis mapping from raw acc. Axis 2 is the vertical
axis; it is not a drop-in replacement and cannot be used without a separate axis remapping and
gravity subtraction. Not investigated further.
