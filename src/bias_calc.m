clear; clc;

calib_data = load("data\calib2_straight.mat").out;
% first 12,000 samples are constant rotation

% figure(1);
% plot(calib_data.GT_position.signals.values(:,1),calib_data.GT_position.signals.values(:,2))
% 
% figure(2);
% gt_quat = calib_data.GT_rotation.signals.values;
% gt_yaw = zeros(size(gt_quat,1), 1);
% for i = 1:size(gt_quat,1)
%     w  = gt_quat(i,1); qx = gt_quat(i,2);
%     qy = gt_quat(i,3); qz = gt_quat(i,4);
%     gt_yaw(i) = atan2(2*(w*qz + qx*qy), 1 - 2*(qy^2 + qz^2));
% end
% plot(gt_yaw);

accel = squeeze(calib_data.Sensor_ACCEL.signals.values);
accel = accel(:, 1:12000);

accel_mean = mean(accel, 2);
accel_std = std(accel,1,2);

gyro = squeeze(calib_data.Sensor_GYRO.signals.values);
gyro = gyro(:, 1:12000);

gyro_mean = mean(gyro, 2);
gyro_std = std(gyro, 1, 2);


calib_data = load("data\calib1_rotate.mat").out;
mag = squeeze(calib_data.Sensor_MAG.signals.values);
mag = mag(:, 1:350);

mag_mean = mean(mag, 2);
mag_std = std(mag, 1, 2);