function [X_Est, P_Est] = myEKF_ca(acc, gyro, mag, ToF1, ToF2, ToF3, Temp, LP_acc) %#ok<INUSD>
%MYEKF_CA  Extended Kalman Filter for a mecanum-wheel robot (2-D + heading).
%          Constant-acceleration system model, persistent-variable interface.
%
% [X_Est, P_Est] = myEKF_ca(acc, gyro, mag, ToF1, ToF2, ToF3, Temp, LP_acc)
%
% Called once per sample from Simulink / a real-time loop.
% Rename file + function to myEKF when replacing the batch version.
%
% ---------- State vector (9 × 1) ----------
%   X = [x; y; theta; vx; vy; omega; ax; ay; b_omega]
%   x, y    : position in arena frame  [m]   (origin at arena centre)
%   theta   : heading                  [rad]
%   vx, vy  : velocity, world frame    [m/s]
%   omega   : yaw rate                 [rad/s]
%   ax, ay  : acceleration, world frame[m/s^2]
%   b_omega : gyro yaw-rate bias       [rad/s]  (residual above static calibration)
%
% ---------- System model ----------
%   Constant-acceleration kinematics; state propagated with fixed dt.
%
% ---------- Sensor inputs ----------
%   acc   [3×1]  accelerometer  – indices 2,3 are robot x,y body axes
%   gyro  [3×1]  gyroscope      – index  1   is yaw rate (vertical axis)
%   mag   [3×1]  magnetometer   – indices 2,3 give heading in xy plane
%   ToF1  [4×1]  right-facing   : ch1 = range [m], ch4 = status (0 = ok)
%   ToF2  [4×1]  forward-facing : ch1 = range [m], ch4 = status (0 = ok)
%   ToF3  [4×1]  left-facing    : ch1 = range [m], ch4 = status (0 = ok)
%   Temp, LP_acc  – ignored
%
% ---------- Arena convention ----------
%   Rectangular arena centred at origin; walls at x = ±Lx, y = ±Ly.
%
% ---------- Board orientation ----------
%   Board is upended so that the robot's horizontal plane maps to sensor
%   indices 2 & 3:
%     acc(2), acc(3)  → body-frame x, y acceleration
%     gyro(1)         → yaw rate (rotation about the vertical axis)
%     mag(2), mag(3)  → heading components

% =========================================================================
%  PERSISTENT STATE
% =========================================================================
persistent X P Q R_imu R_tof initialised
persistent Lx Ly tof_offsets tof_phi
persistent acc_scale gyro_scale mag_declination mag_declination_static dt
persistent chi2_thresh
persistent step tof_update_freq
persistent acc_x_bias acc_y_bias gyro_x_bias gyro_y_bias mag_x_bias mag_y_bias
persistent imu_offset_x imu_offset_y prev_gyro_z vel_damp
persistent tof_prev tof_max_rate zupt_enabled

% =========================================================================
%  ONE-TIME INITIALISATION  ← edit all tuning / geometry here
% =========================================================================
if isempty(initialised)

    % --- Sample period [s] and sensor decimation ratios ------------------
    dt              = 1/200;   % EKF rate  [Hz]
    tof_update_freq = 20;      % ToF  runs at 200/20 = 10 Hz
    step            = 0;

    % --- Arena half-dimensions [m]  (walls at x = ±Lx, y = ±Ly) ----------
    Lx = 1.22;   % total arena 2.44m wide,  walls at x = ±1.22m
    Ly = 1.22;   % total arena 2.44m deep,  walls at y = ±1.22m

    % --- Raw-sensor scaling ----------------------------------------------
    acc_scale       = 1.0;     % multiply raw acc  to get [m/s^2]
    gyro_scale      = 1.09;  % sweep_params: 1.09 best (task2 yaw -10%, pos -0.4% vs 1.10)
    mag_declination        = 1.168;    % calibrated from full rotation dataset [rad]
    mag_declination_static = -1.4245;  % calibrated from calib2_straight static frames [rad]

    % --- ToF sensor offsets from robot centre, body frame [m] ------------
    %   Row i = [dx_fwd, dy_left]
    %     dx_fwd  > 0  →  toward robot front
    %     dy_left > 0  →  toward robot left side
    tof_offsets = [ 0.00,  0.03;   % ToF1 – right-facing
                   -0.02,  0.00;   % ToF2 – forward-facing
                    0.00, -0.03];  % ToF3 – left-facing

    % --- ToF firing angles relative to body forward axis [rad] -----------
    tof_phi = [pi/2; pi; -pi/2];   % right, forward, left (body x points backward)

    % --- Mahalanobis gate  (chi-sq 1 DOF) --------------------------------
    chi2_thresh = 4;   % chi-sq 1 DOF; sweep confirmed 3-5 optimal

    % --- Initial state ---------------------------------------------------
    X = zeros(9, 1);   % assume robot starts at arena centre, at rest, zero bias

    % --- Initial covariance P --------------------------------------------
    P = diag([ 0.50,  0.50,  deg2rad(45), ...   % x, y, theta
               0.50,  0.50,  0.20,        ...   % vx, vy, omega
               1.00,  1.00,  0.05 ].^2);        % ax, ay, b_omega

    % --- Process noise Q -------------------------------------------------
    Q = diag([ 5e-3, 5e-3, deg2rad(3), ...   % x, y, theta
               0.10, 0.10, 0.10,       ...   % vx, vy, omega
               0.25, 0.25, 3e-3 ].^2);       % ax, ay, b_omega (random walk — allows motors-on bias shift)

    % --- IMU measurement noise  z = [acc_bx, acc_by, omega] -------------
    R_imu = diag([ 0.50, 0.50, 0.02 ].^2);

    % --- ToF measurement noise [m^2] (per-sensor: side=ToF1/3, fwd=ToF2) --
    R_tof = [(0.09)^2; (0.04)^2; (0.09)^2];

    % --- Sensor Calibrations ---------------------------------------------
    acc_x_bias = 0.0275;      acc_y_bias = -0.41;
    gyro_x_bias = -0.0072;    gyro_y_bias = -0.0013;
    % Mag biases retained only for step-1 heading initialisation
    mag_x_bias = -5.475113e-05; mag_y_bias = 7.017891e-05;

    % --- IMU offset from robot center of rotation [m, body frame] --------
    % Centre ~20cm from ToFs, IMU ~1cm from ToFs → IMU ~0.19m from centre
    imu_offset_x = 0.15;   % optimal from sweep [-0.25, +0.25]; best net aggregate
    imu_offset_y = 0.00;
    prev_gyro_z  = 0;

    % --- ToF rate-of-change gate -------------------------------------------
    % Reject readings that jump more than this between consecutive samples.
    % At 10Hz and max ~1m/s robot speed, range changes ~0.1m/sample max.
    tof_prev = [nan; nan; nan];   % previous valid reading per sensor
    global SWEEP_TOF_RATE;
    if ~isempty(SWEEP_TOF_RATE)
        tof_max_rate = SWEEP_TOF_RATE;
    else
        tof_max_rate = 0.12;          % max allowed change [m] per ToF sample
    end

    % --- Velocity damping (friction model) ---------------------------------
    % alpha < 1 applies exponential velocity decay: v_next = alpha*v + a*dt
    % Physically: mecanum wheels have friction; robot can't coast indefinitely.
    global SWEEP_VEL_DAMP;
    if ~isempty(SWEEP_VEL_DAMP)
        vel_damp = SWEEP_VEL_DAMP;
    else
        vel_damp = 1.000;   % no damping; sweep showed it hurts task1 without net benefit
    end

    % --- ZUPT (zero-velocity pseudo-measurement when stationary) ----------
    global SWEEP_ZUPT;
    if ~isempty(SWEEP_ZUPT)
        zupt_enabled = SWEEP_ZUPT;
    else
        zupt_enabled = true;
    end

    initialised = true;
end

% =========================================================================
%  STEP COUNTER  (drives sensor decimation)
% =========================================================================
step = step + 1;

do_tof = (step == 1) || (mod(step, tof_update_freq) == 0);   % fire at step 1, then every 20th (10 Hz)

% =========================================================================
%  EXTRACT MEASUREMENTS
% =========================================================================
% Board upended: robot horizontal plane on sensor indices 2 & 3
acc_bx = (-double(acc(2)) - acc_x_bias) * acc_scale;    % body x-acceleration  [m/s^2]
acc_by = (double(acc(3))  - acc_y_bias) * acc_scale;    % body y-acceleration  [m/s^2]
gyro_z = (double(gyro(1)) - gyro_x_bias) * gyro_scale;

% Centripetal correction: subtract IMU-offset-induced acceleration.
% IMU at offset r from center of rotation measures a_true + ω²·r
% (tangential α×r term omitted — finite-differencing gyro at 200Hz is too noisy)
omega_raw = gyro_z;
acc_bx = acc_bx + omega_raw^2 * imu_offset_x;
acc_by = acc_by + omega_raw^2 * imu_offset_y;
fast_spin = abs(gyro_z) > 0.5;
% tof_fast_spin = abs(gyro_z) > 0.75;
tof_fast_spin = 0;

% Magnetometer: one-time heading seed at step 1 only, gated on low gyro_z
% (motors off). During operation, EMI corrupts mag so it is not used further.
if step == 1
    theta_seed = wrapToPi(atan2(double(mag(3)) - mag_y_bias, ...
                                double(mag(2)) - mag_x_bias) + mag_declination_static);
    X(3)    = theta_seed;
    P(3,3)  = deg2rad(10)^2;   % tighten heading uncertainty after seed
end

% ToF: ch1=range [m], ch2=ambient, ch3=signal, ch4=status (0=valid)
tof_d   = double([ToF1(1); ToF2(1); ToF3(1)]);
tof_ok  = double([ToF1(4); ToF2(4); ToF3(4)]) == 0;
tof_sig = double([ToF1(3); ToF2(3); ToF3(3)]);  % return signal strength

% =========================================================================
%  PREDICTION STEP  — constant acceleration
% =========================================================================
x_s = X(1);  y_s = X(2);  th    = X(3);
vx  = X(4);  vy  = X(5);  om    = X(6);
axw = X(7);  ayw = X(8);  b_om  = X(9);

X_p = [
    x_s + vx*dt + 0.5*axw*dt^2;
    y_s + vy*dt + 0.5*ayw*dt^2;
    wrapToPi(th + om*dt);
    vel_damp*vx + axw*dt;
    vel_damp*vy + ayw*dt;
    om;
    axw;
    ayw;
    b_om   % bias is constant between updates (random-walk noise in Q)
];

% State-transition Jacobian  F = ∂f/∂X
F      = eye(9);
F(1,4) = dt;    F(1,7) = 0.5*dt^2;
F(2,5) = dt;    F(2,8) = 0.5*dt^2;
F(3,6) = dt;
F(4,4) = vel_damp;  F(4,7) = dt;
F(5,5) = vel_damp;  F(5,8) = dt;

is_stationary_q = abs(gyro_z) < 0.10 && sqrt(acc_bx^2 + acc_by^2) < 0.15;
if fast_spin
    Q_cur      = Q;
    Q_cur(1,1) = (0.15)^2;
    Q_cur(2,2) = (0.15)^2;
    Q_cur(9,9) = 0;
elseif is_stationary_q
    Q_cur = Q;   % allow bias to drift (track slow EMI changes at rest)
else
    Q_cur      = Q;
    Q_cur(9,9) = 0;   % freeze bias during motion
end
P_p = F * P * F' + Q_cur;

% =========================================================================
%  UPDATE — IMU  (acc + gyro every step)
% =========================================================================
th_p = X_p(3);
axp  = X_p(7);
ayp  = X_p(8);
omp  = X_p(6);
b_p  = X_p(9);

% Predicted body-frame accelerations  (world → body rotation)
h_ax  =  axp*cos(th_p) + ayp*sin(th_p);
h_ay  = -axp*sin(th_p) + ayp*cos(th_p);

% Gyro predicted measurement includes estimated residual bias:
h_gyro = omp + b_p;

% During fast rotation, inflate acc noise so filter ignores acc for translation.
if fast_spin
    R_imu_cur = diag([ 1.0, 1.0, 0.02 ].^2);
else
    R_imu_cur = R_imu;
end

h_imu = [h_ax; h_ay; h_gyro];

H_imu = zeros(3, 9);
H_imu(1,7) =  cos(th_p);   H_imu(1,8) = sin(th_p);
H_imu(2,7) = -sin(th_p);   H_imu(2,8) = cos(th_p);
H_imu(3,6) =  1;            H_imu(3,9) = 1;

z_imu  = [acc_bx; acc_by; gyro_z];
nu_imu = z_imu - h_imu;

S_imu = H_imu * P_p * H_imu' + R_imu_cur;
K_imu = P_p * H_imu' / S_imu;

X_u    = X_p + K_imu * nu_imu;
X_u(3) = wrapToPi(X_u(3));
P_u    = (eye(9) - K_imu * H_imu) * P_p;

% =========================================================================
%  UPDATE — Zero-rotation pseudo-measurement
%  When the robot is stationary, omega_true = 0. Any gyro reading beyond
%  noise is bias. This directly drives b_omega estimation during stops.
% =========================================================================
if is_stationary_q && step > 1
    % Zero-rotation: omega should be zero
    H_zr    = zeros(1, 9);
    H_zr(6) = 1;            % observes omega state
    R_zr    = (0.01)^2;

    nu_zr = 0 - X_u(6);
    S_zr  = H_zr * P_u * H_zr' + R_zr;
    K_zr  = P_u * H_zr' / S_zr;

    X_u = X_u + K_zr * nu_zr;
    P_u = (eye(9) - K_zr * H_zr) * P_u;

    % Zero-velocity: vx, vy should be zero when stationary
    if zupt_enabled && sqrt(X_u(4)^2 + X_u(5)^2) < 0.05
        H_zv      = zeros(2, 9);
        H_zv(1,4) = 1;
        H_zv(2,5) = 1;
        R_zv      = diag([0.01, 0.01].^2);

        nu_zv = [0; 0] - X_u(4:5);
        S_zv  = H_zv * P_u * H_zv' + R_zv;
        K_zv  = P_u * H_zv' / S_zv;

        X_u = X_u + K_zv * nu_zv;
        P_u = (eye(9) - K_zv * H_zv) * P_u;
    end
end

% =========================================================================
%  UPDATE — ToF sensors  (10 Hz: every 20th step)
% =========================================================================
if tof_fast_spin
    X_u(4) = 0; X_u(5) = 0; % Zero velocity
end
for s = 1:3 * do_tof * ~tof_fast_spin
    if ~tof_ok(s)
        continue
    end

    % Rate-of-change gate: reject readings that jump too much
    if ~isnan(tof_prev(s)) && abs(tof_d(s) - tof_prev(s)) > tof_max_rate
        tof_prev(s) = tof_d(s);   % update anyway for next comparison
        continue
    end
    tof_prev(s) = tof_d(s);

    th_u    = X_u(3);
    dx_body = tof_offsets(s, 1);
    dy_body = tof_offsets(s, 2);

    % Sensor position in world frame
    sx = X_u(1) + cos(th_u)*dx_body - sin(th_u)*dy_body;
    sy = X_u(2) + sin(th_u)*dx_body + cos(th_u)*dy_body;

    % Predicted range and analytical Jacobian w.r.t. sensor pos & angle
    [h_pred, dh_dsx, dh_dsy, ~] = tof_measurement( ...
        sx, sy, th_u, tof_phi(s), Lx, Ly);

    if h_pred <= 0
        continue
    end

    % Reject readings whose predicted hit point is near an arena corner.
    ray_world = th_u + tof_phi(s);
    hit_x = sx + h_pred * cos(ray_world);
    hit_y = sy + h_pred * sin(ray_world);
    corner_margin = 0.05;
    if abs(hit_x) > (Lx - corner_margin) && abs(hit_y) > (Ly - corner_margin)
        continue
    end

    H_tof    = zeros(1, 9);
    H_tof(1) = dh_dsx;
    H_tof(2) = dh_dsy;

    % Incidence-angle-adaptive R_tof (floor 0.30, max ~11× inflation)
    if abs(dh_dsx) > 1e-9
        inc_cos = max(abs(cos(ray_world)), 0.30);
    else
        inc_cos = max(abs(sin(ray_world)), 0.30);
    end
    R_tof_a = R_tof(s) / inc_cos^2;

    nu_tof = tof_d(s) - h_pred;
    S_tof  = H_tof * P_u * H_tof' + R_tof_a;

    if (nu_tof^2 / S_tof) > chi2_thresh && step ~= 1
        continue
    end

    K_tof = P_u * H_tof' / S_tof;
    X_u   = X_u + K_tof * nu_tof;
    X_u(3) = wrapToPi(X_u(3));
    P_u   = (eye(9) - K_tof * H_tof) * P_u;
end

% =========================================================================
%  ARENA BOUNDARY CLAMP
%  Robot cannot be outside the walls — clamp prevents h_pred≤0 which would
%  skip all ToF updates and let the filter run open-loop indefinitely.
% =========================================================================
X_u(1) = max(-Lx + 0.05, min(Lx - 0.05, X_u(1)));
X_u(2) = max(-Ly + 0.05, min(Ly - 0.05, X_u(2)));

% =========================================================================
%  STORE STATE AND RETURN
% =========================================================================
X = X_u;
P = P_u;

X_Est = X;
P_Est = P;

end % myEKF_ca

% =========================================================================
%  LOCAL HELPER — analytical ToF range measurement model
% =========================================================================
function [h, dh_dsx, dh_dsy, dh_dth_s] = tof_measurement(sx, sy, th, phi_s, Lx, Ly)
%TOF_MEASUREMENT  Range from sensor world position (sx,sy) to nearest wall.
%
%   Arena walls at x = ±Lx, y = ±Ly (origin at centre).
%   Ray fired at world angle (th + phi_s).
%   Returns h (predicted range) and partial derivatives for EKF Jacobian.

    ray = th + phi_s;
    cd  = cos(ray);
    sd  = sin(ray);

    t = inf(4, 1);
    if abs(cd) > 1e-9
        t(1) = ( Lx - sx) / cd;   % wall x = +Lx
        t(2) = (-Lx - sx) / cd;   % wall x = -Lx
    end
    if abs(sd) > 1e-9
        t(3) = ( Ly - sy) / sd;   % wall y = +Ly
        t(4) = (-Ly - sy) / sd;   % wall y = -Ly
    end

    t(t <= 1e-6) = inf;            % discard behind-sensor intersections

    % Reject if two walls are nearly equidistant: wall assignment is fragile
    % and a small heading error flips it, causing a discontinuous h_pred jump.
    t_valid = sort(t(t < inf));
    if length(t_valid) >= 2 && t_valid(2) < 1.20 * t_valid(1)
        h = -1;  dh_dsx = 0;  dh_dsy = 0;  dh_dth_s = 0;
        return
    end

    [h, wall] = min(t);

    if isinf(h)
        h = -1;  dh_dsx = 0;  dh_dsy = 0;  dh_dth_s = 0;
        return
    end

    % Analytical partial derivatives (depend on which wall was hit)
    switch wall
        case 1  % x = +Lx
            dh_dsx   = -1 / cd;
            dh_dsy   =  0;
            dh_dth_s =  ( Lx - sx) * sd / cd^2;
        case 2  % x = -Lx
            dh_dsx   = -1 / cd;
            dh_dsy   =  0;
            dh_dth_s =  (-Lx - sx) * sd / cd^2;
        case 3  % y = +Ly
            dh_dsx   =  0;
            dh_dsy   = -1 / sd;
            dh_dth_s = -( Ly - sy) * cd / sd^2;
        case 4  % y = -Ly
            dh_dsx   =  0;
            dh_dsy   = -1 / sd;
            dh_dth_s = -(-Ly - sy) * cd / sd^2;
        otherwise
            h = -1;  dh_dsx = 0;  dh_dsy = 0;  dh_dth_s = 0;
    end
end
