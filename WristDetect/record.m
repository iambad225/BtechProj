% --- 1. DATA RECORDER ---
clear; clc;
portName = 'COM3'; % CHANGE THIS
baudRate = 115200;

% Setup
label = input('Enter label (rest/clench): ', 's');
duration = 10; % Record for 10 seconds

try
    s = serialport(portName, baudRate); 
    configureTerminator(s, "LF");
    flush(s);
catch
    error('Connect ESP32 first!');
end

disp(['Recording ' label ' data in 2 seconds... GET READY!']);
pause(2);
disp('GO! (Recording...)');

% Recording Loop
data = [];
tic;
while toc < duration
    if s.NumBytesAvailable > 0
        val = str2double(readline(s));
        if ~isnan(val)
            data = [data; val];
        end
    end
end

% Save to CSV
csvwrite([label '_data.csv'], data);
disp(['Saved ' num2str(length(data)) ' samples to ' label '_data.csv']);
clear s;