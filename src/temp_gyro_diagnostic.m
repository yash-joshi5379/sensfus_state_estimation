%% temp_gyro_diagnostic.m
% Check correlation between temperature sensor and gyro bias drift.
% Saves one plot per dataset: Temp and gyro_z (bias-subtracted) over time.

gyro_x_bias = -0.0112;
gyro_scale  = 1.1;

datasets = { ...
    'data\task1_1 1.mat',  'task1_1' ; ...
    'data\task1_2 1.mat',  'task1_2' ; ...
    'data\task1_3.mat',    'task1_3' ; ...
    'data\task2_1 1.mat',  'task2_1' ; ...
    'data\task2_2 1.mat',  'task2_2' ; ...
    'data\task2_3 1.mat',  'task2_3' };

for d = 1:size(datasets, 1)
    raw  = load(datasets{d,1}).out;
    TAG  = datasets{d,2};

    gyro = squeeze(raw.Sensor_GYRO.signals.values)';
    temp = squeeze(raw.Sensor_Temp.signals.values);

    gyro_z = (double(gyro(:,1)) - gyro_x_bias) * gyro_scale;

    % Correlation between temp and gyro_z
    r = corrcoef(temp, gyro_z);
    fprintf('%s: Temp range [%.2f, %.2f]  gyro_z range [%.4f, %.4f]  corr=%.4f\n', ...
        TAG, min(temp), max(temp), min(gyro_z), max(gyro_z), r(1,2));

    % Plot
    fh = figure('Visible', 'off');
    t = (0:length(temp)-1) / 200;

    yyaxis left;
    plot(t, temp, 'b');
    ylabel('Temperature [raw]');

    yyaxis right;
    plot(t, gyro_z, 'r', 'LineWidth', 0.5);
    ylabel('gyro\_z [rad/s]');

    xlabel('Time [s]');
    title(['Temp vs gyro\_z — ' TAG]);
    saveas(fh, [TAG '_temp_gyro.jpg']);
    close(fh);
end
