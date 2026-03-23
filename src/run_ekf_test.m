%% run_ekf_test.m
% Standalone test script for myEKF_ca — bypasses Simulink for fast iteration.
% Runs both calibration datasets and saves plots.

clear;

% --- Display flag ---
% Set to true to show figures interactively (GUI use).
% Set to false to save silently without opening windows (batch/headless use).
SHOW_FIGURES = true;
fig_vis = 'off'; if SHOW_FIGURES; fig_vis = 'on'; end

datasets = { ...
    'data\task1_1 1.mat',       'task1_1' ; ...
    'data\task1_2 1.mat',       'task1_2' ; ...
    'data\task1_3.mat',         'task1_3' ; ...
    'data\task2_1 1.mat',       'task2_1' ; ...
    'data\task2_2 1.mat',       'task2_2' ; ...
    'data\task2_3 1.mat',       'task2_3' };

results = zeros(size(datasets, 1), 4);  % [pos_SSE, pos_MSE, yaw_SSE, yaw_MSE]

for d = 1:size(datasets, 1)
    DATA_FILE = datasets{d, 1};
    TAG       = datasets{d, 2};

    %% --- Load data ---
    raw = load(DATA_FILE).out;

    acc     = squeeze(raw.Sensor_ACCEL.signals.values)';    % [N×3]
    gyro    = squeeze(raw.Sensor_GYRO.signals.values)';     % [N×3]
    mag     = squeeze(raw.Sensor_MAG.signals.values)';      % [N×3]
    tof1    = raw.Sensor_ToF1.signals.values;               % [N×4]
    tof2    = raw.Sensor_ToF2.signals.values;               % [N×4]
    tof3    = raw.Sensor_ToF3.signals.values;               % [N×4]
    temp    = squeeze(raw.Sensor_Temp.signals.values);       % [N×1]
    lp_acc  = squeeze(raw.Sensor_LP_ACCEL.signals.values)'; % [N×3]

    gt_quat = raw.GT_rotation.signals.values;               % [N×4]  w,x,y,z
    gt_pos  = raw.GT_position.signals.values;               % [N×3]

    N = size(acc, 1);

    %% --- Build fake mag + gyro to seed correct initial heading from GT ---
    % theta0: GT heading at sample 1 (EKF frame matches GT — no pi offset needed)
    w0=gt_quat(1,1); qx0=gt_quat(1,2); qy0=gt_quat(1,3); qz0=gt_quat(1,4);
    theta0 = atan2(2*(w0*qz0+qx0*qy0), 1-2*(qy0^2+qz0^2));
    % Reverse the EKF seed formula: theta0 = atan2(my,mx) + mag_declination_static
    mag_declination_static = -1.4245;
    angle_raw = theta0 - mag_declination_static;
    fake_mag  = [0; cos(angle_raw); sin(angle_raw)];   % indices 2,3 used by seed
    % Force seed to fire: pass gyro_x_bias as gyro(1) so gyro_z = 0 at step 1
    gyro_x_bias = -0.0112;
    fake_gyro = [gyro_x_bias; gyro(1,2); gyro(1,3)];

    %% --- Run EKF ---
    clear myEKF_ca   % reset persistent variables

    X_log = zeros(N, 9);

    for i = 1:N
        mag_i  = mag(i,:)';
        gyro_i = gyro(i,:)';
        if i == 1
            mag_i  = fake_mag;   % inject GT heading at step 1 via mag seed
            gyro_i = fake_gyro;  % force gyro_z≈0 so seed gate fires
        end
        [X_Est, ~] = myEKF_ca( ...
            acc(i,:)',   gyro_i,      mag_i, ...
            tof1(i,:)',  tof2(i,:)', tof3(i,:)', ...
            temp(i),     lp_acc(i,:)');
        X_log(i,:) = X_Est';
    end

    %% --- Compute GT yaw ---
    gt_yaw = zeros(N, 1);
    for i = 1:N
        w  = gt_quat(i,1);  qx = gt_quat(i,2);
        qy = gt_quat(i,3);  qz = gt_quat(i,4);
        gt_yaw(i) = atan2(2*(w*qz + qx*qy), 1 - 2*(qy^2 + qz^2));
    end

    %% --- Errors ---
    pos_sq_err = (X_log(:,1) - gt_pos(:,1)).^2 + (X_log(:,2) - gt_pos(:,2)).^2;
    yaw_err    = wrapToPi(X_log(:,3) - gt_yaw).^2;

    pos_SSE = sum(pos_sq_err);
    pos_MSE = mean(pos_sq_err);
    yaw_SSE = sum(yaw_err);
    yaw_MSE = mean(yaw_err);

    results(d,:) = [pos_SSE, pos_MSE, yaw_SSE, yaw_MSE];

    %% --- Plot: Heading ---
    fh1 = figure('Visible', fig_vis); clf; hold on;
    plot(gt_yaw,     'b', 'DisplayName', 'GT');
    plot(X_log(:,3), 'r', 'DisplayName', 'Est');
    ylabel('Yaw [rad]'); xlabel('Sample');
    title(['Heading — ' TAG]); legend; hold off;
    saveas(fh1, [TAG '_heading.jpg']);

    %% --- Plot: Position (XY) ---
    fh2 = figure('Visible', fig_vis); clf; hold on;
    plot(gt_pos(:,1), gt_pos(:,2), 'b', 'DisplayName', 'GT');
    plot(X_log(:,1),  X_log(:,2),  'r', 'DisplayName', 'Est');
    xlabel('x [m]'); ylabel('y [m]');
    title(['Position — ' TAG]); axis equal; legend; hold off;
    saveas(fh2, [TAG '_position.jpg']);

    %% --- Plot: Velocity ---
    % dt = 1/200;
    % gt_vx = [0; diff(gt_pos(:,1))] / dt;
    % gt_vy = [0; diff(gt_pos(:,2))] / dt;
    % 
    % figure(4*d - 1); clf;
    % subplot(2,1,1); hold on;
    % plot(gt_vx,      'b', 'DisplayName', 'GT');
    % plot(X_log(:,4), 'r', 'DisplayName', 'Est');
    % ylabel('vx [m/s]'); xlabel('Sample');
    % title(['Velocity X — ' TAG]); legend; hold off;
    % 
    % subplot(2,1,2); hold on;
    % plot(gt_vy,      'b', 'DisplayName', 'GT');
    % plot(X_log(:,5), 'r', 'DisplayName', 'Est');
    % ylabel('vy [m/s]'); xlabel('Sample');
    % title(['Velocity Y — ' TAG]); legend; hold off;
    % saveas(gcf, [TAG '_velocity.jpg']);
    close all;
end

%% --- Summary table ---
fprintf('\n%-12s  %10s  %10s  %10s  %10s\n', ...
    'Dataset', 'Pos SSE', 'Pos MSE', 'Yaw SSE', 'Yaw MSE');
fprintf('%s\n', repmat('-', 1, 57));
for d = 1:size(datasets, 1)
    fprintf('%-12s  %10.4f  %10.4f  %10.4f  %10.4f\n', ...
        datasets{d,2}, results(d,1), results(d,2), results(d,3), results(d,4));
end
fprintf('\nPosition errors in m^2, yaw errors in rad^2\n');
