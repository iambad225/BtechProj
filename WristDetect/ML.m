% --- Low-Latency Real-Time EMG with Recursive Smoothing ---
clear; clc; close all;

% 1. CONFIGURATION
portName = 'COM3';   
baudRate = 115200;
alpha = 0.2;         % SMOOTHING FACTOR: 0.1 (Heavy/Slow) to 0.9 (Light/Fast)

% 2. CLEANUP
delete(serialportfind);

% 3. SETUP CONNECTION
s = serialport(portName, baudRate);
configureTerminator(s, "LF"); 
flush(s);

% 4. PLOT SETUP
figure('Name', 'Zero-Lag EMG Stream', 'Color', 'w');
hLine = animatedline('Color', 'b', 'LineWidth', 1.5);
ax = gca;
ax.YLim = [0 4095];
xlabel('Time (s)');
grid on;

% 5. INITIALIZE FILTER
filteredVal = 2048; % Start at midpoint
startTimer = tic;
timeWindow = 5;    % Smaller window = less CPU load for plotting

% 6. THE FAST LOOP
try
    while true
        % Check if data is actually there to avoid waiting (blocking)
        if s.NumBytesAvailable > 0
            
            % READ ONLY THE LATEST PACKET
            % If the buffer is full of old data, flush it to stay "live"
            if s.NumBytesAvailable > 100
                flush(s); 
                continue; % Skip this cycle to catch up to the newest data
            end
            
            dataLine = readline(s);
            rawVal = str2double(dataLine);
            
            if ~isnan(rawVal)
                % RECURSIVE FILTER (Exponential Moving Average)
                % No buffer/array needed = Zero processing lag
                filteredVal = (alpha * rawVal) + ((1 - alpha) * filteredVal);
                
                currentTime = toc(startTimer);
                
                % Update Graph
                addpoints(hLine, currentTime, filteredVal);
                ax.XLim = [currentTime - timeWindow, currentTime];
                
                % Use 'limitrate' to keep the GUI responsive
                drawnow limitrate;
            end
        end
    end
catch
    clear s;
    disp('Port Closed.');
end