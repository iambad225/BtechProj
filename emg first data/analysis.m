clc; clear; close all;

slow = readtable('slow.csv');
fast = readtable('fast.csv');
weight = readtable('weight.csv');

set = fast;

rawData = set.Raw_EMG;
envelope = set.Envelope;
time = set.Timestamp;


%%
fs = 420;

n = length(rawData);
window = hamming(n); % Using Hamming window for better PSD estimation
[pxx, f] = pwelch(rawData, window, n/2, [], fs);

figure(1);
semilogx(f, pxx);
title('Power Spectral Density of Raw EMG Data');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

%%
fc = 5;
[b, a] = butter(4, fc/(fs/2), 'low');

cleanedData = filtfilt(b, a, rawData);

fast.cleaned = cleanedData;

figure(2);
subplot(3, 1, 1)
plot(rawData);
title('Raw Data')
grid on

subplot(3, 1, 2)
plot(cleanedData);
title('Cleaned data')
grid on

subplot(3, 1, 3)
plot(envelope);
title("Envelope")
grid on