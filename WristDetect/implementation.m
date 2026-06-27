% --- 3. REAL-TIME LOGISTIC REGRESSION (SMOOTH + ROBUST) ---
clear; clc; close all;

% 1. Load Parameters
if exist('trainedLogReg.mat', 'file')
    load('trainedLogReg.mat'); 
else
    error('trainedLogReg.mat not found! Run the Trainer first.');
end

% 2. Setup Serial
portName = 'COM3'; % Check your Device Manager!
baudRate = 115200;

delete(serialportfind); % Clean up old connections
try
    s = serialport(portName, baudRate);
    configureTerminator(s, "LF");
    flush(s);
    disp(['Connected to ', portName]);
catch
    error('Could not connect to ESP32. Close Arduino Monitor.');
end

% 3. Plot Setup
figure('Name', 'Live Smooth Detection', 'Color', 'w');
hLine = animatedline('Color', [0 0.447 0.741], 'LineWidth', 1.5); % Professional Blue
ax = gca; 
ax.YLim = [0 4095];
ax.XGrid = 'on';
title('Live Logistic Regression (with Smoothing & Hysteresis)');
xlabel('Time (s)'); ylabel('Signal Amplitude');

% 4. PARAMETERS (TUNE THESE)
% A. Sensitivity Control (Hysteresis)
clenchThresh  = 0.85;  % HARD Trigger: Must be 85% sure to clench
releaseThresh = 0.40;  % Release Trigger: Must drop below 40% to release

% B. Graph Smoothing Factor (0.01 = Heavy smooth, 1.0 = No smooth)
alpha = 0.2; 

% 5. Loop Initialization
buffer = zeros(50, 1);
smoothedVal = 2000; % Start at approx midpoint
isClenched = false;
startTimer = tic;
timeWindow = 5;

disp('Running... (Sensitivity reduced: Requires strong clench)');

while true
    if s.NumBytesAvailable > 0
        % Read Data
        dataLine = readline(s);
        newVal = str2double(dataLine);
        
        if ~isnan(newVal)
            % Update Buffer for ML (Keep raw data for accurate math)
            buffer = [buffer(2:end); newVal];
            
            % --- A. GRAPH SMOOTHING (EMA) ---
            % Smooths the visual line ONLY (Does not delay detection)
            smoothedVal = (alpha * newVal) + ((1 - alpha) * smoothedVal);
            
            % --- B. ML CALCULATION ---
            currentRMS = sqrt(mean(buffer.^2));
            z = Intercept + (Coeff_RMS * currentRMS);
            probability = 1 / (1 + exp(-z));
            
            % --- C. ROBUST DECISION LOGIC (Hysteresis) ---
            if ~isClenched
                % To turn ON, we need HIGH confidence (0.85)
                if probability > clenchThresh
                    set(ax, 'Color', [1 0.8 0.8]); % Red Background
                    fprintf('[%.2f] CLENCH DETECTED (Prob: %.2f)\n', toc(startTimer), probability);
                    isClenched = true;
                end
            else
                % To turn OFF, signal must drop significantly (below 0.40)
                % This ignores small dips in muscle power during a hold
                if probability < releaseThresh
                    set(ax, 'Color', 'w'); % White Background
                    fprintf('[%.2f] Released (Prob: %.2f)\n', toc(startTimer), probability);
                    isClenched = false;
                end
            end
            
            % Update Plot
            currentTime = toc(startTimer);
            addpoints(hLine, currentTime, smoothedVal); % Plotting SMOOTH value
            ax.XLim = [currentTime - timeWindow, currentTime];
            
            % Anti-Lag Flush
            if s.NumBytesAvailable > 100, flush(s); end
            
            drawnow limitrate;
        end
    end
end