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
plot(wrapToPi(gt_yaw + pi));
plot(out.P_Est.signals.values);
legend(["GT", "Est"]);
hold off;

% RMSE calculation
gt_x = sensorLog.GT_position.signals.values(:,1);
gt_y = sensorLog.GT_position.signals.values(:,2);
est_x = out.X_Est.signals.values(:,1);
est_y = out.X_Est.signals.values(:,2);
est_th = out.P_Est.signals.values(:);
gt_th = wrapToPi(gt_yaw + pi);

n = min([length(gt_x), length(est_x)]);
pos_err = sqrt((est_x(1:n) - gt_x(1:n)).^2 + (est_y(1:n) - gt_y(1:n)).^2);
x_err = est_x(1:n) - gt_x(1:n);
y_err = est_y(1:n) - gt_y(1:n);
th_err = wrapToPi(est_th(1:n) - gt_th(1:n));

fprintf('\n--- RMSE Results ---\n');
fprintf('Position RMSE:  %.4f m\n', sqrt(mean(pos_err.^2)));
fprintf('X RMSE:         %.4f m\n', sqrt(mean(x_err.^2)));
fprintf('Y RMSE:         %.4f m\n', sqrt(mean(y_err.^2)));
fprintf('Heading RMSE:   %.4f rad (%.2f deg)\n', sqrt(mean(th_err.^2)), rad2deg(sqrt(mean(th_err.^2))));
fprintf('Max pos error:  %.4f m\n', max(pos_err));