clc; clear;
model = "src\IKS02A1_VL53L1A1_Both_ROS_fromworkspace_2025b_forstudents.slx";
load_system(model);

%% Load data and run
sensorLog = load("data\task2_2 1.mat").out;
% Clear initial NaN values
sensorLog.GT_time.signals.values(isnan(sensorLog.GT_time.signals.values)) = 0;
%%
out = sim(model);

figure(1); clf; hold on;
plot(sensorLog.GT_position.signals.values(:,1),sensorLog.GT_position.signals.values(:,2))
plot(out.X_Est.signals.values(:,1),out.X_Est.signals.values(:,2))
legend(["GT", "Est"]);
hold off;

figure(2); clf; hold on;
gt_quat = sensorLog.GT_rotation.signals.values;
gt_yaw = zeros(size(gt_quat,1), 1);
for i = 1:size(gt_quat,1)
    w  = gt_quat(i,1); qx = gt_quat(i,2);
    qy = gt_quat(i,3); qz = gt_quat(i,4);
    gt_yaw(i) = atan2(2*(w*qz + qx*qy), 1 - 2*(qy^2 + qz^2));
end
plot(gt_yaw);
plot(out.X_Est.signals.values(:,3));
legend(["GT", "Est"]);
hold off;