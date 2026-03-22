clc; clear;

% All dataset files
files = {
    'task1_1 1', 'task1_1';
    'task1_2 1', 'task1_2';
    'task1_3',   'task1_3';
    'task2_1 1', 'task2_1';
    'task2_2 1', 'task2_2';
    'task2_3 1', 'task2_3';
    'task2_4',   'task2_4';
};

fprintf('%-12s %12s %12s %12s %12s\n', 'Dataset', 'Pos SSE', 'Pos MSE', 'Yaw SSE', 'Yaw MSE');
fprintf('%s\n', repmat('-', 1, 62));

for f = 1:size(files, 1)
    % Load data
    sensorLog = load(fullfile('data', [files{f,1} '.mat'])).out;
    sensorLog.GT_time.signals.values(isnan(sensorLog.GT_time.signals.values)) = 0;
    
    % Run simulation
    clear myEKF
    simout = sim('IKS02A1_VL53L1A1_Both_ROS_fromworkspace_2025b_forstudents');
    
    % Extract GT
    gt_x = sensorLog.GT_position.signals.values(:,1);
    gt_y = sensorLog.GT_position.signals.values(:,2);
    gt_quat = sensorLog.GT_rotation.signals.values;
    gt_yaw = zeros(size(gt_quat,1), 1);
    for i = 1:size(gt_quat,1)
        w = gt_quat(i,1); qx = gt_quat(i,2);
        qy = gt_quat(i,3); qz = gt_quat(i,4);
        gt_yaw(i) = atan2(2*(w*qz + qx*qy), 1 - 2*(qy^2 + qz^2));
    end
    gt_th = wrapToPi(gt_yaw + pi);
    
    % Extract estimates
    est_x = simout.X_Est.signals.values(:,1);
    est_y = simout.X_Est.signals.values(:,2);
    est_th = simout.P_Est.signals.values(:);
    
    % Compute errors
    n = min([length(gt_x), length(est_x)]);
    pos_se = (est_x(1:n) - gt_x(1:n)).^2 + (est_y(1:n) - gt_y(1:n)).^2;
    yaw_se = wrapToPi(est_th(1:n) - gt_th(1:n)).^2;
    
    pos_sse = sum(pos_se);
    pos_mse = mean(pos_se);
    yaw_sse = sum(yaw_se);
    yaw_mse = mean(yaw_se);
    
    fprintf('%-12s %12.4f %12.4f %12.4f %12.4f\n', files{f,2}, pos_sse, pos_mse, yaw_sse, yaw_mse);
end