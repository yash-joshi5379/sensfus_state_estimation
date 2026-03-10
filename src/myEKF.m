% Your function should take in the logged struct and output your state
% estimate as well your state covariance and the ground truth (X,Y,Z).
% [X_Est, P_Est, GT] = myEKF(out)
%
% State vector: [x, y, theta, bias_omega_z]
%   x, y          : 2D position of robot centre in arena frame [m]
%   theta         : yaw angle [rad]
%   bias_omega_z  : gyroscope z-axis bias [rad/s]
%
% Sensors used:
%   - ISM330DHCX accelerometer + gyroscope (104 Hz) : process model
%   - IIS2MDC magnetometer (50 Hz)                  : yaw correction
%   - VL53L1X ToF x3 (10 Hz)                        : position correction
%
% Arena: rectangular, walls at x = +/-Lx, y = +/-Ly (origin at centre)
%
% ToF sensor body-frame layout (measure from robot centre):
%   ToF1 (left)    : offset [dx1, dy1], firing angle phi1
%   ToF2 (forward) : offset [dx2, dy2], firing angle phi2
%   ToF3 (right)   : offset [dx3, dy3], firing angle phi3

function [X_Est, P_Est, GT] = myEKF(out)

% --- Arena dimensions (half-widths) [m] ---
% Replace with actual values when provided
Lx = 2.44; % +- 0.01
Ly = 2.44; % +- 0.01

% --- ToF sensor mounting angles [rad] relative to robot body x-axis ---
% phi1: left sensor fires left (+90 deg)
% phi2: forward sensor fires forward (0 deg)
% phi3: right sensor fires right (-90 deg)
phi = [pi/2, 0, -pi/2];

% --- ToF sensor offsets from robot centre in body frame [m] ---
% Format: [forward_offset, lateral_offset]
% Positive forward = towards robot front, positive lateral = towards robot left
% MEASURE THESE FROM THE PHYSICAL ROBOT
tof_offsets = [ 0.20,  0.05;   % ToF1: left sensor   (forward, left)
                0.15,  0.00;   % ToF2: forward sensor (forward, centre)
                0.20, -0.05];  % ToF3: right sensor   (forward, right)

% --- Extract raw data from Simulink struct ---
% IMU data: stored as [1x3xN], squeeze and transpose to [Nx3]
gyro_data  = squeeze(out.Sensor_GYRO.signals.values)';
accel_data = squeeze(out.Sensor_ACCEL.signals.values)';
mag_data   = squeeze(out.Sensor_MAG.signals.values)';

% ToF data: stored as [Nx4] -> columns: [distance, ambient, signal, status]
tof1_data  = out.Sensor_ToF1.signals.values;
tof2_data  = out.Sensor_ToF2.signals.values;
tof3_data  = out.Sensor_ToF3.signals.values;

% Timestamps for each sensor
imu_time  = out.Sensor_GYRO.time;
mag_time  = out.Sensor_MAG.time;
tof1_time = out.Sensor_ToF1.time;
tof2_time = out.Sensor_ToF2.time;
tof3_time = out.Sensor_ToF3.time;

% Ground truth position and orientation
gt_pos  = out.GT_position.signals.values;
gt_quat = out.GT_rotation.signals.values;
gt_time = squeeze(out.GT_time.signals.values);

% --- Convert ground truth quaternion to yaw angle ---
gt_yaw = zeros(size(gt_quat,1), 1);
for i = 1:size(gt_quat,1)
    w  = gt_quat(i,1); qx = gt_quat(i,2);
    qy = gt_quat(i,3); qz = gt_quat(i,4);
    gt_yaw(i) = atan2(2*(w*qz + qx*qy), 1 - 2*(qy^2 + qz^2));
end

% GT output: [x, y, yaw]
GT = [gt_pos(:,1), gt_pos(:,2), gt_yaw];

% --- Initialise state from first ground truth sample ---
x_est = [gt_pos(1,1); gt_pos(1,2); gt_yaw(1); 0.0];

% Estimate initial gyro bias from first 50 IMU samples (near-stationary start)
n_init = min(50, size(gyro_data,1));
x_est(4) = mean(gyro_data(1:n_init, 3));

% Initial state covariance
P = diag([0.01, 0.01, 0.01, 0.001]);

% --- Noise covariance parameters ---
% Process noise Q: models unmodelled acceleration and bias drift
Q = diag([0.001, 0.001, 0.001, 0.0001]);

% Measurement noise R
R_mag = 0.05;   % magnetometer yaw noise [rad^2]
R_tof = 0.05;   % ToF distance noise [m^2]

% Mahalanobis gating threshold: chi-squared 1 DOF at 99th percentile
% Rejects ToF readings inconsistent with current state uncertainty (e.g. wall holes)
chi2_thresh = 6.63;

% --- Preallocate output arrays ---
N = length(imu_time);
X_Est = zeros(N, 4);
P_Est = zeros(N, 4, 4);

% Indices tracking next unprocessed sample for lower-rate sensors
mag_idx = 1;
tof_idx = 1;

% --- Main EKF loop running at IMU rate (104 Hz) ---
for k = 1:N
    t_now = imu_time(k);

    if k == 1
        dt = 1/104;
    else
        dt = imu_time(k) - imu_time(k-1);
    end

    % ---- PREDICT STEP ----
    % Bias-corrected yaw rate
    omega_z = gyro_data(k,3) - x_est(4);

    % Body-frame accelerations
    ax = accel_data(k,1);
    ay = accel_data(k,2);
    th = x_est(3);

    % Rotate body-frame acceleration into world frame
    ax_w = ax*cos(th) - ay*sin(th);
    ay_w = ax*sin(th) + ay*cos(th);

    % Propagate state using constant-acceleration kinematics
    x_est(1) = x_est(1) + 0.5*ax_w*dt^2;
    x_est(2) = x_est(2) + 0.5*ay_w*dt^2;
    x_est(3) = wrapToPi(x_est(3) + omega_z*dt);
    % bias_omega_z propagates as constant (process noise handles drift)

    % Linearised Jacobian of process model F = df/dx
    F = eye(4);
    F(1,3) = -0.5*(ax*sin(th) + ay*cos(th))*dt^2;
    F(2,3) =  0.5*(ax*cos(th) - ay*sin(th))*dt^2;
    F(3,4) = -dt;

    % Covariance prediction
    P = F*P*F' + Q;

    % ---- MAGNETOMETER UPDATE (50 Hz) ----
    % atan2(my, mx) gives absolute yaw reference, correcting gyro drift
    while mag_idx <= length(mag_time) && mag_time(mag_idx) <= t_now
        mx = mag_data(mag_idx,1);
        my = mag_data(mag_idx,2);
        yaw_mag = atan2(my, mx);

        H = [0, 0, 1, 0];
        innov = wrapToPi(yaw_mag - x_est(3));
        S = H*P*H' + R_mag;
        K = P*H'/S;
        x_est = x_est + K*innov;
        x_est(3) = wrapToPi(x_est(3));
        P = (eye(4) - K*H)*P;
        mag_idx = mag_idx + 1;
    end

    % ---- TOF UPDATE (10 Hz) ----
    % Each sensor processed independently with status gate + Mahalanobis gate
    if tof_idx <= size(tof1_data,1) && tof1_time(tof_idx) <= t_now
        tof_batch = [tof1_data(tof_idx,:);
                     tof2_data(tof_idx,:);
                     tof3_data(tof_idx,:)];

        for s = 1:3
            dist_meas = tof_batch(s,1);
            status    = tof_batch(s,4);

            % Status byte: 0 = valid, nonzero = hardware error or out-of-range
            if status ~= 0
                continue;
            end

            % Sensor position in world frame: rotate body offset by current yaw
            dx_body = tof_offsets(s,1);
            dy_body = tof_offsets(s,2);
            sx = x_est(1) + cos(th)*dx_body - sin(th)*dy_body;
            sy = x_est(2) + sin(th)*dx_body + cos(th)*dy_body;

            % Predicted wall distance fired from sensor world position
            [h_pred, dh_dsx, dh_dsy, dh_dth_s] = tof_measurement( ...
                sx, sy, x_est(3), phi(s), Lx, Ly);

            if h_pred <= 0
                continue;
            end

            % Chain rule: dh/d[x,y,theta] accounting for sensor offset
            % dsx/dx=1, dsx/dtheta = -sin(th)*dx - cos(th)*dy
            % dsy/dy=1, dsy/dtheta =  cos(th)*dx - sin(th)*dy
            dsx_dth = -sin(th)*dx_body - cos(th)*dy_body;
            dsy_dth =  cos(th)*dx_body - sin(th)*dy_body;

            dh_dx  = dh_dsx;
            dh_dy  = dh_dsy;
            dh_dth = dh_dsx*dsx_dth + dh_dsy*dsy_dth + dh_dth_s;

            H = [dh_dx, dh_dy, dh_dth, 0];
            innov = dist_meas - h_pred;
            S = H*P*H' + R_tof;

            % Mahalanobis gate: reject readings beyond chi2 threshold
            if (innov^2 / S) > chi2_thresh
                continue;
            end

            % EKF update
            K = P*H'/S;
            x_est = x_est + K*innov;
            x_est(3) = wrapToPi(x_est(3));
            P = (eye(4) - K*H)*P;
        end

        tof_idx = tof_idx + 1;
    end

    X_Est(k,:)   = x_est';
    P_Est(k,:,:) = P;
end

% --- RMSE Calculation ---
% Remove duplicate GT timestamps before interpolation
[gt_time_u, uid] = unique(gt_time);
gt_x_interp   = interp1(gt_time_u, gt_pos(uid,1), imu_time, 'linear', 'extrap');
gt_y_interp   = interp1(gt_time_u, gt_pos(uid,2), imu_time, 'linear', 'extrap');
gt_yaw_interp = interp1(gt_time_u, gt_yaw(uid),   imu_time, 'linear', 'extrap');

% Position RMSE: Euclidean distance error at each timestep
pos_err  = sqrt((X_Est(:,1) - gt_x_interp).^2 + (X_Est(:,2) - gt_y_interp).^2);
rmse_pos = sqrt(mean(pos_err.^2));

% Yaw RMSE: wrap angle difference to [-pi, pi] before squaring
yaw_err  = wrapToPi(X_Est(:,3) - gt_yaw_interp);
rmse_yaw = sqrt(mean(yaw_err.^2));

fprintf('--- EKF Performance ---\n');
fprintf('Position RMSE : %.4f m\n', rmse_pos);
fprintf('Yaw RMSE      : %.4f rad (%.2f deg)\n', rmse_yaw, rad2deg(rmse_yaw));

end

% -----------------------------------------------------------------------
% tof_measurement: ToF ray-rectangle measurement model
%
% Inputs:
%   sx, sy  : sensor position in world frame [m] (NOT robot centre)
%   th      : robot yaw [rad]
%   phi_s   : sensor firing angle relative to body x-axis [rad]
%   Lx, Ly  : arena half-dimensions [m]
%
% Outputs:
%   h           : predicted distance to nearest wall [m]
%   dh_dsx      : partial derivative w.r.t. sensor x position
%   dh_dsy      : partial derivative w.r.t. sensor y position
%   dh_dth_s    : partial derivative w.r.t. theta (ray direction only)
%
% Note: full dh/dtheta in main loop combines dh_dth_s with offset Jacobian
%
% Method: fire ray from sensor world position at world angle (th+phi_s),
% intersect with all four walls, take minimum positive distance.
% -----------------------------------------------------------------------
function [h, dh_dsx, dh_dsy, dh_dth_s] = tof_measurement(sx, sy, th, phi_s, Lx, Ly)

    ray_angle = th + phi_s;
    cd = cos(ray_angle);
    sd = sin(ray_angle);

    % Distance to each wall along the ray direction
    t_candidates = inf(4,1);
    if abs(cd) > 1e-9
        t_candidates(1) = ( Lx - sx) / cd;  % right wall  x = +Lx
        t_candidates(2) = (-Lx - sx) / cd;  % left wall   x = -Lx
    end
    if abs(sd) > 1e-9
        t_candidates(3) = ( Ly - sy) / sd;  % top wall    y = +Ly
        t_candidates(4) = (-Ly - sy) / sd;  % bottom wall y = -Ly
    end

    % Keep only forward intersections
    t_candidates(t_candidates <= 1e-6) = inf;
    [h, wall_idx] = min(t_candidates);

    if isinf(h)
        h = -1; dh_dsx = 0; dh_dsy = 0; dh_dth_s = 0;
        return;
    end

    % Analytical Jacobian: form depends on which wall the ray hits
    switch wall_idx
        case 1  % right wall x = +Lx
            dh_dsx   = -1/cd;
            dh_dsy   =  0;
            dh_dth_s =  (Lx - sx)*sd / cd^2;
        case 2  % left wall x = -Lx
            dh_dsx   = -1/cd;
            dh_dsy   =  0;
            dh_dth_s =  (-Lx - sx)*sd / cd^2;
        case 3  % top wall y = +Ly
            dh_dsx   =  0;
            dh_dsy   = -1/sd;
            dh_dth_s = -(Ly - sy)*cd / sd^2;
        case 4  % bottom wall y = -Ly
            dh_dsx   =  0;
            dh_dsy   = -1/sd;
            dh_dth_s = -(-Ly - sy)*cd / sd^2;
    end
end