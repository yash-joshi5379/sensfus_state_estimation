%% run_ekf_test.m
% Standalone test script for myEKF_ca — bypasses Simulink for fast iteration.
% Runs both calibration datasets and saves plots.

clc; clear;

datasets = { ...
    'data\calib1_rotate.mat',   'rotate'; ...
    'data\calib2_straight.mat', 'straight'; ...
    'data\task2_1 1.mat',       'task2_1' ; ...
    'data\task2_2 1.mat',       'task2_2' };

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

    %% --- Run EKF ---
    clear myEKF_ca   % reset persistent variables

    X_log = zeros(N, 8);
    t_start = tic;

    for i = 1:N
        [X_Est, ~] = myEKF_ca( ...
            acc(i,:)',   gyro(i,:)',  mag(i,:)', ...
            tof1(i,:)',  tof2(i,:)', tof3(i,:)', ...
            temp(i),     lp_acc(i,:)');
        X_log(i,:) = X_Est';
    end

    fprintf('[%s] EKF ran %d steps in %.2f s (%.0f Hz effective)\n', ...
        TAG, N, toc(t_start), N/toc(t_start));

    %% --- Compute GT yaw ---
    gt_yaw = zeros(N, 1);
    for i = 1:N
        w  = gt_quat(i,1);  qx = gt_quat(i,2);
        qy = gt_quat(i,3);  qz = gt_quat(i,4);
        gt_yaw(i) = atan2(2*(w*qz + qx*qy), 1 - 2*(qy^2 + qz^2));
    end

    %% --- Plot: Heading ---
    figure(2*d - 1); clf; hold on;
    plot(gt_yaw,     'b', 'DisplayName', 'GT');
    plot(X_log(:,3), 'r', 'DisplayName', 'Est');
    ylabel('Yaw [rad]'); xlabel('Sample');
    title(['Heading — ' TAG]); legend; hold off;
    saveas(gcf, [TAG '_heading.jpg']);

    %% --- Plot: Position (XY) ---
    figure(2*d); clf; hold on;
    plot(gt_pos(:,1), gt_pos(:,2), 'b', 'DisplayName', 'GT');
    plot(X_log(:,1),  X_log(:,2),  'r', 'DisplayName', 'Est');
    xlabel('x [m]'); ylabel('y [m]');
    title(['Position — ' TAG]); axis equal; legend; hold off;
    saveas(gcf, [TAG '_position.jpg']);
end
