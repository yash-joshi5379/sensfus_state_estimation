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
    % 'data\task1_1 1.mat',       'task1_1' ; ...
    % 'data\task1_2 1.mat',       'task1_2' ; ...
    % 'data\task1_3.mat',         'task1_3' ; ...
    % 'data\task1_4.mat',         'task1_4' ; ...
    'data\task2_1 1.mat',       'task2_1' ; ...
    'data\task2_2 1.mat',       'task2_2' ; ...
    'data\task2_3 1.mat',       'task2_3' ; ...
    'data\task2_4.mat',         'task2_4' ...
    };

results = zeros(size(datasets, 1), 5);  % [pos_SSE, pos_RMSE, yaw_SSE, yaw_RMSE, final_pos_err]

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
    w0=gt_quat(3,1); qx0=gt_quat(3,2); qy0=gt_quat(3,3); qz0=gt_quat(3,4);
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

    final_pos_err = sqrt((X_log(N,1) - gt_pos(N,1))^2 + (X_log(N,2) - gt_pos(N,2))^2);
    results(d,:) = [pos_SSE, sqrt(pos_MSE), yaw_SSE, sqrt(yaw_MSE), final_pos_err];

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

    %% --- Plot: Per-step error stem plots ---
    t_s = (0:N-1) / 200;   % time axis in seconds
    pos_err_step = sqrt((X_log(:,1) - gt_pos(:,1)).^2 + (X_log(:,2) - gt_pos(:,2)).^2);
    yaw_err_step = abs(wrapToPi(X_log(:,3) - gt_yaw));

    fh3 = figure('Visible', fig_vis); clf;

    subplot(2,1,1);
    stem(t_s, pos_err_step, 'filled', 'MarkerSize', 1, 'Color', [0.2 0.5 0.8]);
    ylabel('Position error (m)');
    title(['Per-step errors — ' TAG]);
    grid on;

    subplot(2,1,2);
    stem(t_s, yaw_err_step, 'filled', 'MarkerSize', 1, 'Color', [0.8 0.3 0.2]);
    ylabel('Heading error (rad)');
    xlabel('Time (s)');
    grid on;

    saveas(fh3, [TAG '_errors.jpg']);

    %% --- Plot: Velocity ---
    dt = 1/200;
    gt_vx = [0; diff(gt_pos(:,1))] / dt;
    gt_vy = [0; diff(gt_pos(:,2))] / dt;

    fh4 = figure('Visible', fig_vis); clf;
    subplot(2,1,1); hold on;
    plot(t_s, gt_vx,      'b', 'DisplayName', 'GT');
    plot(t_s, X_log(:,4), 'r', 'DisplayName', 'Est');
    ylabel('vx [m/s]'); xlabel('Time (s)');
    title(['Velocity X — ' TAG]); legend; hold off;

    subplot(2,1,2); hold on;
    plot(t_s, gt_vy,      'b', 'DisplayName', 'GT');
    plot(t_s, X_log(:,5), 'r', 'DisplayName', 'Est');
    ylabel('vy [m/s]'); xlabel('Time (s)');
    title(['Velocity Y — ' TAG]); legend; hold off;
    saveas(fh4, [TAG '_velocity.jpg']);
    if ~SHOW_FIGURES; close all; end
end

%% --- Summary table ---
fprintf('\n%-12s  %10s  %10s  %10s  %10s  %12s\n', ...
    'Dataset', 'Pos SSE', 'Pos RMSE', 'Yaw SSE', 'Yaw RMSE', 'Final Pos Err');
fprintf('%s\n', repmat('-', 1, 72));
for d = 1:size(datasets, 1)
    fprintf('%-12s  %10.4f  %10.4f  %10.4f  %10.4f  %12.4f\n', ...
        datasets{d,2}, results(d,1), results(d,2), results(d,3), results(d,4), results(d,5));
end
fprintf('\nPosition errors in m^2, yaw errors in rad^2, final pos error in m\n');
