%% run_ekf_test_gt.m
% EKF test with GT heading locked at every step.
% Isolates position estimation error from heading error — shows the RMSE
% floor achievable with perfect orientation knowledge.
% Uses the same ToF model, parameters and dataset list as run_ekf_test.m.

clear;

SHOW_FIGURES = false;
fig_vis = 'off'; if SHOW_FIGURES; fig_vis = 'on'; end

datasets = { ...
    'data\task1_1 1.mat',   'task1_1' ; ...
    'data\task1_2 1.mat',   'task1_2' ; ...
    'data\task1_3.mat',     'task1_3' ; ...
    'data\task1_4.mat',     'task1_4' ; ...
    'data\task2_1 1.mat',   'task2_1' ; ...
    'data\task2_2 1.mat',   'task2_2' ; ...
    'data\task2_3 1.mat',   'task2_3' ; ...
    'data\task2_4.mat',     'task2_4' };

% --- Parameters (must match myEKF_ca.m) ---
dt              = 1/200;
tof_update_freq = 20;
Lx = 1.22;  Ly = 1.22;
acc_scale    = 1.0;
acc_x_bias   = 0.0275;   acc_y_bias = -0.41;
gyro_x_bias  = -0.0112;
tof_offsets  = [ 0.00,  0.03;
                -0.02,  0.00;
                 0.00, -0.03];
tof_phi      = [pi/2; pi; -pi/2];
chi2_thresh  = 4.0;
R_tof        = (0.07)^2;
corner_margin = 0.1;
wall_ambig_thresh = 1.20;

% State: [x; y; theta; vx; vy; omega; ax; ay; b_omega] — same as myEKF_ca
% theta is overridden by GT after every predict+IMU step.
R_imu = diag([ 0.50, 0.50, 0.02 ].^2);
Q = diag([ 5e-3, 5e-3, deg2rad(3), ...
           0.10, 0.10, 0.10,       ...
           0.25, 0.25, 3e-3 ].^2);

results = zeros(size(datasets,1), 4);

for d = 1:size(datasets,1)
    DATA_FILE = datasets{d,1};
    TAG       = datasets{d,2};

    raw = load(DATA_FILE).out;
    acc    = squeeze(raw.Sensor_ACCEL.signals.values)';
    gyro   = squeeze(raw.Sensor_GYRO.signals.values)';
    tof1   = raw.Sensor_ToF1.signals.values;
    tof2   = raw.Sensor_ToF2.signals.values;
    tof3   = raw.Sensor_ToF3.signals.values;
    gt_quat = raw.GT_rotation.signals.values;
    gt_pos  = raw.GT_position.signals.values;

    N = size(acc,1);

    % --- GT yaw at every step ---
    gt_yaw = zeros(N,1);
    for i = 1:N
        w=gt_quat(i,1); qx=gt_quat(i,2); qy=gt_quat(i,3); qz=gt_quat(i,4);
        gt_yaw(i) = atan2(2*(w*qz+qx*qy), 1-2*(qy^2+qz^2));
    end

    % --- Initialise state from GT ---
    X = zeros(9,1);
    X(1) = gt_pos(1,1);
    X(2) = gt_pos(1,2);
    X(3) = gt_yaw(1);
    P = diag([ 0.50, 0.50, deg2rad(45), ...
               0.50, 0.50, 0.20,        ...
               1.00, 1.00, 0.05 ].^2);

    X_log = zeros(N,9);
    step  = 0;

    for i = 1:N
        step = step + 1;
        do_tof = (mod(step, tof_update_freq) == 0);

        % Extract IMU
        acc_bx = -double(acc(i,2)) * acc_scale - acc_x_bias;
        acc_by =  double(acc(i,3)) * acc_scale - acc_y_bias;
        gyro_z = (double(gyro(i,1)) - gyro_x_bias) * 1.1;
        fast_spin = abs(gyro_z) > 0.5;

        % --- PREDICT ---
        x_s=X(1); y_s=X(2); th=X(3);
        vx=X(4);  vy=X(5);  om=X(6);
        axw=X(7); ayw=X(8); b_om=X(9);

        X_p = [ x_s + vx*dt + 0.5*axw*dt^2;
                y_s + vy*dt + 0.5*ayw*dt^2;
                wrapToPi(th + om*dt);
                vx + axw*dt;
                vy + ayw*dt;
                om;  axw;  ayw;  b_om ];

        F = eye(9);
        F(1,4)=dt; F(1,7)=0.5*dt^2;
        F(2,5)=dt; F(2,8)=0.5*dt^2;
        F(3,6)=dt; F(4,7)=dt; F(5,8)=dt;

        is_stationary = abs(gyro_z) < 0.10 && sqrt(acc_bx^2+acc_by^2) < 0.15;
        Q_cur = Q;
        if fast_spin
            Q_cur(1,1)=(0.15)^2; Q_cur(2,2)=(0.15)^2; Q_cur(9,9)=0;
        elseif is_stationary
            % allow bias drift
        else
            Q_cur(9,9)=0;
        end
        P_p = F*P*F' + Q_cur;

        % --- IMU UPDATE ---
        th_p=X_p(3); axp=X_p(7); ayp=X_p(8); omp=X_p(6); b_p=X_p(9);
        if fast_spin
            R_imu_cur = diag([1.0,1.0,0.02].^2);
        else
            R_imu_cur = R_imu;
        end
        h_ax  =  axp*cos(th_p) + ayp*sin(th_p);
        h_ay  = -axp*sin(th_p) + ayp*cos(th_p);
        h_gyro = omp + b_p;
        H_imu = zeros(3,9);
        H_imu(1,7)= cos(th_p); H_imu(1,8)= sin(th_p);
        H_imu(2,7)=-sin(th_p); H_imu(2,8)= cos(th_p);
        H_imu(3,6)=1; H_imu(3,9)=1;
        z_imu = [acc_bx; acc_by; gyro_z];
        nu_imu = z_imu - [h_ax;h_ay;h_gyro];
        S_imu = H_imu*P_p*H_imu' + R_imu_cur;
        K_imu = P_p*H_imu'/S_imu;
        X_u    = X_p + K_imu*nu_imu;
        X_u(3) = wrapToPi(X_u(3));
        P_u    = (eye(9)-K_imu*H_imu)*P_p;

        % --- ZERO-ROTATION UPDATE (stationary) ---
        if is_stationary && step > 1
            H_zr=zeros(1,9); H_zr(6)=1; R_zr=(0.01)^2;
            nu_zr=0-X_u(6);
            S_zr=H_zr*P_u*H_zr'+R_zr;
            K_zr=P_u*H_zr'/S_zr;
            X_u=X_u+K_zr*nu_zr;
            P_u=(eye(9)-K_zr*H_zr)*P_u;
        end

        % --- GT HEADING LOCK: replace estimated theta with GT ---
        % Very tight update so position estimation sees the true heading.
        theta_gt = gt_yaw(i);
        H_gt    = zeros(1,9); H_gt(3) = 1;
        R_gt    = (deg2rad(0.5))^2;   % effectively locks theta to GT
        nu_gt   = wrapToPi(theta_gt - X_u(3));
        S_gt    = H_gt*P_u*H_gt' + R_gt;
        K_gt    = P_u*H_gt'/S_gt;
        X_u     = X_u + K_gt*nu_gt;
        X_u(3)  = wrapToPi(X_u(3));
        P_u     = (eye(9)-K_gt*H_gt)*P_u;

        % --- TOF UPDATES ---
        tof_d  = double([tof1(i,1); tof2(i,1); tof3(i,1)]);
        tof_ok = double([tof1(i,4); tof2(i,4); tof3(i,4)]) == 0;

        for s = 1:3 * do_tof
            if ~tof_ok(s); continue; end
            th_u = X_u(3);
            dx_body=tof_offsets(s,1); dy_body=tof_offsets(s,2);
            sx = X_u(1)+cos(th_u)*dx_body-sin(th_u)*dy_body;
            sy = X_u(2)+sin(th_u)*dx_body+cos(th_u)*dy_body;
            [h_pred,dh_dsx,dh_dsy,~] = gt_tof_measurement(sx,sy,th_u,tof_phi(s),Lx,Ly);
            if h_pred <= 0; continue; end
            ray_world = th_u + tof_phi(s);
            hit_x = sx + h_pred*cos(ray_world);
            hit_y = sy + h_pred*sin(ray_world);
            if abs(hit_x)>(Lx-corner_margin) && abs(hit_y)>(Ly-corner_margin); continue; end
            if abs(dh_dsx)>1e-9
                inc_cos=max(abs(cos(ray_world)),0.30);
            else
                inc_cos=max(abs(sin(ray_world)),0.30);
            end
            R_tof_a = R_tof/inc_cos^2;
            nu_tof = tof_d(s)-h_pred;
            H_tof=zeros(1,9); H_tof(1)=dh_dsx; H_tof(2)=dh_dsy;
            S_tof=H_tof*P_u*H_tof'+R_tof_a;
            if (nu_tof^2/S_tof)>chi2_thresh; continue; end
            K_tof=P_u*H_tof'/S_tof;
            X_u=X_u+K_tof*nu_tof;
            X_u(3)=wrapToPi(X_u(3));
            P_u=(eye(9)-K_tof*H_tof)*P_u;
        end

        % Arena boundary clamp
        X_u(1)=max(-Lx+0.05,min(Lx-0.05,X_u(1)));
        X_u(2)=max(-Ly+0.05,min(Ly-0.05,X_u(2)));

        X=X_u; P=P_u;
        X_log(i,:)=X';
    end

    % --- Errors ---
    pos_sq_err = (X_log(:,1)-gt_pos(:,1)).^2+(X_log(:,2)-gt_pos(:,2)).^2;
    yaw_err    = wrapToPi(X_log(:,3)-gt_yaw).^2;
    pos_SSE = sum(pos_sq_err);
    pos_RMSE = sqrt(mean(pos_sq_err));
    yaw_SSE = sum(yaw_err);
    yaw_RMSE = sqrt(mean(yaw_err));
    results(d,:) = [pos_SSE, pos_RMSE, yaw_SSE, yaw_RMSE];

    % --- Plots ---
    fh1 = figure('Visible',fig_vis); clf; hold on;
    plot(gt_yaw,     'b','DisplayName','GT');
    plot(X_log(:,3), 'r','DisplayName','Est');
    ylabel('Yaw [rad]'); xlabel('Sample');
    title(['Heading (GT-locked) — ' TAG]); legend; hold off;
    saveas(fh1,[TAG '_gt_heading.jpg']);

    fh2 = figure('Visible',fig_vis); clf; hold on;
    plot(gt_pos(:,1),gt_pos(:,2),'b','DisplayName','GT');
    plot(X_log(:,1), X_log(:,2), 'r','DisplayName','Est');
    xlabel('x [m]'); ylabel('y [m]');
    title(['Position (GT-locked) — ' TAG]); axis equal; legend; hold off;
    saveas(fh2,[TAG '_gt_position.jpg']);

    if ~SHOW_FIGURES; close all; end
end

fprintf('\n%-12s  %10s  %10s  %10s  %10s\n', ...
    'Dataset','Pos SSE','Pos RMSE','Yaw SSE','Yaw RMSE');
fprintf('%s\n',repmat('-',1,57));
for d = 1:size(datasets,1)
    fprintf('%-12s  %10.4f  %10.4f  %10.4f  %10.4f\n', ...
        datasets{d,2},results(d,1),results(d,2),results(d,3),results(d,4));
end
fprintf('\nPosition errors in m^2/m, yaw errors in rad^2/rad\n');

% =========================================================================
%  LOCAL HELPER — same ray-wall model as myEKF_ca
% =========================================================================
function [h, dh_dsx, dh_dsy, dh_dth_s] = gt_tof_measurement(sx,sy,th,phi_s,Lx,Ly)
    ray=th+phi_s; cd=cos(ray); sd=sin(ray);
    t=inf(4,1);
    if abs(cd)>1e-9; t(1)=(Lx-sx)/cd; t(2)=(-Lx-sx)/cd; end
    if abs(sd)>1e-9; t(3)=(Ly-sy)/sd; t(4)=(-Ly-sy)/sd; end
    t(t<=1e-6)=inf;
    t_valid=sort(t(t<inf));
    if length(t_valid)>=2 && t_valid(2)<1.20*t_valid(1)
        h=-1; dh_dsx=0; dh_dsy=0; dh_dth_s=0; return
    end
    [h,wall]=min(t);
    if isinf(h); h=-1; dh_dsx=0; dh_dsy=0; dh_dth_s=0; return; end
    switch wall
        case 1; dh_dsx=-1/cd; dh_dsy=0;    dh_dth_s= (Lx-sx)*sd/cd^2;
        case 2; dh_dsx=-1/cd; dh_dsy=0;    dh_dth_s=(-Lx-sx)*sd/cd^2;
        case 3; dh_dsx=0;    dh_dsy=-1/sd; dh_dth_s=-(Ly-sy)*cd/sd^2;
        case 4; dh_dsx=0;    dh_dsy=-1/sd; dh_dth_s=-(-Ly-sy)*cd/sd^2;
        otherwise; h=-1; dh_dsx=0; dh_dsy=0; dh_dth_s=0;
    end
end
