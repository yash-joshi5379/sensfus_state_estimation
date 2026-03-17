clc; clear;
sensorLog = load("data\calib2_straight.mat").out;

sensorLog.GT_time.signals.values(isnan(sensorLog.GT_time.signals.values)) = 0;