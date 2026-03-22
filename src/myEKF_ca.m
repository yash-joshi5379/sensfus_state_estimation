function [X_Est, P_Est] = myEKF_ca(acc, gyro, mag, ToF1, ToF2, ToF3, Temp, LP_acc) %#ok<INUSD>
%MYEKF_CA  Extended Kalman Filter for a mecanum-wheel robot (2-D + heading).
%          Constant-acceleration system model, persistent-variable interface.
%
% [X_Est, P_Est] = myEKF_ca(acc, gyro, mag, ToF1, ToF2, ToF3, Temp, LP_acc)
%
% Called once per sample from Simulink / a real-time loop.
% Rename file + function to myEKF when replacing the batch version.
%
% ---------- State vector (8 × 1) ----------
%   X = [x; y; theta; vx; vy; omega; ax; ay]
%   x, y    : position in arena frame  [m]   (origin at arena centre)
%   theta   : heading                  [rad]
%   vx, vy  : velocity, world frame    [m/s]
%   omega   : yaw rate                 [rad/s]
%   ax, ay  : acceleration, world frame[m/s^2]
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
    gyro_scale      = 1.02;   % reduced from 1.05: task datasets show ~3% over-integration per revolution
    mag_declination        = 1.168;    % calibrated from full rotation dataset [rad]
    mag_declination_static = -1.4245;  % calibrated from calib2_straight static frames [rad]

    % --- ToF sensor offsets from robot centre, body frame [m] ------------
    %   Row i = [dx_fwd, dy_left]
    %     dx_fwd  > 0  →  toward robot front
    %     dy_left > 0  →  toward robot left side
    tof_offsets = [ 0.20,  0.04;   % ToF1 – right-facing
                    0.209, 0.00;   % ToF2 – forward-facing
                    0.20, -0.04];  % ToF3 – left-facing

    % --- ToF firing angles relative to body forward axis [rad] -----------
    tof_phi = [-pi/2; 0; pi/2];   % right, forward, left

    % --- Mahalanobis gate  (chi-sq 1 DOF, 99th percentile) --------------
    chi2_thresh = 6.63;

    % --- Initial state ---------------------------------------------------
    X = zeros(8, 1);   % assume robot starts at arena centre, at rest

    % --- Initial covariance P --------------------------------------------
    P = diag([ 0.50,  0.50,  deg2rad(45), ...   % x, y, theta
               0.50,  0.50,  0.20,        ...   % vx, vy, omega
               1.00,  1.00 ].^2);               % ax, ay

    % --- Process noise Q -------------------------------------------------
    Q = diag([ 5e-3, 5e-3, deg2rad(3), ...   % x, y, theta
               0.05, 0.05, 0.10,       ...   % vx, vy, omega
               0.30, 0.30 ].^2);             % ax, ay

    % --- IMU measurement noise  z = [acc_bx, acc_by, omega] -------------
    R_imu = diag([ 0.50, 0.50, 0.02 ].^2);

    % --- ToF measurement noise [m^2] -------------------------------------
    R_tof = (0.05)^2;

    % --- Sensor Calibrations ---------------------------------------------
    acc_x_bias = 0.0275;      acc_y_bias = -0.3963;
    gyro_x_bias = -0.0112;    gyro_y_bias = -0.0013;
    % Mag biases retained only for step-1 heading initialisation
    mag_x_bias = -5.475113e-05; mag_y_bias = 7.017891e-05;

    initialised = true;
end

% =========================================================================
%  STEP COUNTER  (drives sensor decimation)
% =========================================================================
step = step + 1;

do_tof = (mod(step, tof_update_freq) == 0);   % true every 20th call (10 Hz)

% =========================================================================
%  EXTRACT MEASUREMENTS
% =========================================================================
% Board upended: robot horizontal plane on sensor indices 2 & 3
acc_bx = double(acc(2))  * acc_scale - acc_x_bias;    % body x-acceleration  [m/s^2]
acc_by = double(acc(3))  * acc_scale - acc_y_bias;    % body y-acceleration  [m/s^2]
gyro_z   = (double(gyro(1)) - gyro_x_bias) * gyro_scale;  % bias before scale
fast_spin = abs(gyro_z) > 0.5;

% Magnetometer is not used for updates — hard-iron interference from motors
% is ~180 deg during operation (measured on calibration data).
% One-time heading seed at step 1: only when robot is stationary (motors off,
% gyro near zero), so mag is reliable. Skipped automatically if already moving.
if step == 1 && abs(gyro_z) < 0.15
    X(3)    = wrapToPi(atan2(double(mag(3)) - mag_y_bias, ...
                             double(mag(2)) - mag_x_bias) + mag_declination_static);
    P(3,3)  = (0.20)^2;   % tighter than default 45° — mag reliable with motors off
end

% ToF: channel 1 = range [m], channel 4 = status (0 = valid)
tof_d  = double([ToF1(1); ToF2(1); ToF3(1)]);
tof_ok = double([ToF1(4); ToF2(4); ToF3(4)]) == 0;

% =========================================================================
%  PREDICTION STEP  — constant acceleration
% =========================================================================
x_s = X(1);  y_s = X(2);  th  = X(3);
vx  = X(4);  vy  = X(5);  om  = X(6);
axw = X(7);  ayw = X(8);

X_p = [
    x_s + vx*dt + 0.5*axw*dt^2;
    y_s + vy*dt + 0.5*ayw*dt^2;
    wrapToPi(th + om*dt);
    vx  + axw*dt;
    vy  + ayw*dt;
    om;
    axw;
    ayw
];

% State-transition Jacobian  F = ∂f/∂X
F      = eye(8);
F(1,4) = dt;    F(1,7) = 0.5*dt^2;
F(2,5) = dt;    F(2,8) = 0.5*dt^2;
F(3,6) = dt;
F(4,7) = dt;
F(5,8) = dt;

% During fast spin, inflate position process noise so P grows faster.
% This widens the chi2 gate for ToF — valid range measurements are accepted
% even if position has drifted slightly due to centripetal contamination.
if fast_spin
    Q_cur      = Q;
    Q_cur(1,1) = (0.15)^2;
    Q_cur(2,2) = (0.15)^2;
else
    Q_cur = Q;
end
P_p = F * P * F' + Q_cur;

% =========================================================================
%  UPDATE — IMU  (acc + gyro every step;  magnetometer at 50 Hz)
% =========================================================================
th_p = X_p(3);
axp  = X_p(7);
ayp  = X_p(8);
omp  = X_p(6);

% During fast rotation, centripetal acceleration contaminates the body-frame
% acc measurement and projects into world-frame ax/ay → position drift.
% Inflate acc noise channels so the filter ignores acc for translation when spinning.
if fast_spin
    R_imu_cur = diag([ 5.0, 5.0, 0.02 ].^2);   % distrust acc, keep gyro
else
    R_imu_cur = R_imu;
end

% Predicted body-frame accelerations  (world → body rotation)
h_ax  =  axp*cos(th_p) + ayp*sin(th_p);
h_ay  = -axp*sin(th_p) + ayp*cos(th_p);

% --- acc + gyro update (every step, no mag) ---
h_imu = [h_ax; h_ay; omp];

H_imu = zeros(3, 8);
H_imu(1,3) = 0;                                  % acc does not correct theta (gyro handles it)
H_imu(1,7) =  cos(th_p);                         % ∂h_ax/∂ax
H_imu(1,8) =  sin(th_p);                         % ∂h_ax/∂ay
H_imu(2,3) = 0;                                  % acc does not correct theta (gyro handles it)
H_imu(2,7) = -sin(th_p);                         % ∂h_ay/∂ax
H_imu(2,8) =  cos(th_p);                         % ∂h_ay/∂ay
H_imu(3,6) =  1;                                 % ∂omega/∂omega

z_imu  = [acc_bx; acc_by; gyro_z];
nu_imu = z_imu - h_imu;

S_imu = H_imu * P_p * H_imu' + R_imu_cur;
K_imu = P_p * H_imu' / S_imu;

X_u    = X_p + K_imu * nu_imu;
X_u(3) = wrapToPi(X_u(3));
P_u    = (eye(8) - K_imu * H_imu) * P_p;

% =========================================================================
%  UPDATE — ToF sensors  (10 Hz: every 20th step)
% =========================================================================
for s = 1:3 * do_tof
    if ~tof_ok(s)
        continue
    end

    th_u    = X_u(3);
    dx_body = tof_offsets(s, 1);
    dy_body = tof_offsets(s, 2);

    % Sensor position in world frame
    sx = X_u(1) + cos(th_u)*dx_body - sin(th_u)*dy_body;
    sy = X_u(2) + sin(th_u)*dx_body + cos(th_u)*dy_body;

    % Predicted range and analytical Jacobian w.r.t. sensor pos & angle
    [h_pred, dh_dsx, dh_dsy, dh_dth_s] = tof_measurement( ...
        sx, sy, th_u, tof_phi(s), Lx, Ly);

    if h_pred <= 0
        continue
    end

    % ∂(sensor world pos)/∂theta via offset rotation
    dsx_dth = -sin(th_u)*dx_body - cos(th_u)*dy_body;
    dsy_dth =  cos(th_u)*dx_body - sin(th_u)*dy_body;

    % Full measurement Jacobian row  H_tof [1 × 8]
    % H_tof(3): full ∂h/∂theta — sensor position rotates with theta (first two
    % terms) plus ray direction rotates with theta (dh_dth_s).
    % With correct arena dimensions and working position estimate, ToF residuals
    % now reflect true heading error rather than position error.
    H_tof    = zeros(1, 8);
    H_tof(1) = dh_dsx;
    H_tof(2) = dh_dsy;
    H_tof(3) = 0;   % heading excluded: dh/dtheta can be several m/rad at oblique angles;
                    % tiny R_tof means any position residual drives enormous heading spikes.

    nu_tof = tof_d(s) - h_pred;
    S_tof  = H_tof * P_u * H_tof' + R_tof;

    % Mahalanobis gate — rejects outliers (holes in walls, reflections, etc.)
    if (nu_tof^2 / S_tof) > chi2_thresh
        continue
    end

    K_tof    = P_u * H_tof' / S_tof;
    K_tof(3) = 0;   % zero to match H(3)=0 and maintain P symmetry
    X_u      = X_u + K_tof * nu_tof;
    X_u(3) = wrapToPi(X_u(3));
    P_u    = (eye(8) - K_tof * H_tof) * P_u;
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
