clc; clear;
sensorLog = load("data\task1_1 1.mat").out;

sensorLog.GT_time.signals.values(isnan(sensorLog.GT_time.signals.values)) = 0;