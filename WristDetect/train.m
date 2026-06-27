% --- 2. OPTIMIZED SVM TRAINER ---
clear; clc; close all;

% A. Load Data
try
    restRaw = csvread('rest_data.csv');
    clenchRaw = csvread('clench_data.csv');
catch
    error('Run Step 1 first to generate CSV files!');
end

% B. Enhanced Feature Extraction
% Adding Waveform Length (WL) is key for distinguishing muscle bursts from noise
windowSize = 50; 
extract = @(x) [ ...
    sqrt(mean(x.^2)), ...           % Feature 1: RMS (Power)
    mean(abs(x - mean(x))), ...     % Feature 2: MAV (Intensity)
    sum(abs(diff(x))) ...           % Feature 3: WL (Complexity/Jaggedness)
];

features = []; labels = [];

% Process Data
for i = 1:windowSize:(length(restRaw)-windowSize)
    features = [features; extract(restRaw(i:i+windowSize-1))];
    labels = [labels; -1];
end
for i = 1:windowSize:(length(clenchRaw)-windowSize)
    features = [features; extract(clenchRaw(i:i+windowSize-1))];
    labels = [labels; 1];
end

% C. Train Non-Linear SVM (RBF Kernel)
% Fixed the Struct Error by removing 'Display' from the options
disp('Optimizing SVM parameters... please wait.');
SVMModel = fitcsvm(features, labels, ...
    'KernelFunction', 'rbf', ...
    'Standardize', true, ...
    'OptimizeHyperparameters', 'auto', ... 
    'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations', 15));

% D. Export Parameters
save('trainedSVM.mat', 'SVMModel');

fprintf('\n=== TRAINING COMPLETE ===\n');
fprintf('Model saved as trainedSVM.mat\n');

% E. Visualization (3D Feature Space)
figure('Color', 'w');
scatter3(features(labels==-1,1), features(labels==-1,2), features(labels==-1,3), 40, 'b', 'filled'); hold on;
scatter3(features(labels==1,1), features(labels==1,2), features(labels==1,3), 40, 'r', 'filled');
xlabel('RMS (Power)'); ylabel('MAV (Intensity)'); zlabel('Waveform Length');
legend('Rest', 'Clench'); title('Optimized 3D Feature Space'); grid on; view(45, 30);