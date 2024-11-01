% Constants
INPUT_LENGTH = 10;
WEIGHT_LENGTH = 4;
NUM_ANTENNAS = WEIGHT_LENGTH;
FREQUENCY = 5;  % Hz
PI = 3.14159265359;
STARTTIME = 0;
ENDTIME = 99;

% Generate time array
time = linspace(STARTTIME, ENDTIME, INPUT_LENGTH);

% Generate input signals for each antenna
signals = zeros(NUM_ANTENNAS, INPUT_LENGTH);
for i = 1:NUM_ANTENNAS
    phase_shift = (i - 1) * PI / 4;  % Different phase for each antenna
    for j = 1:INPUT_LENGTH
        signals(i, j) = sin(2 * PI * FREQUENCY * time(j) + phase_shift);
    end
end

% Define weights for alignment
weights = 1:NUM_ANTENNAS;  % Example weights [1, 2, 3, 4]

% Beamforming: Sum signals with phase alignment
beamformed_signal = zeros(1, INPUT_LENGTH);
for j = 1:INPUT_LENGTH
    for i = 1:NUM_ANTENNAS
        beamformed_signal(j) = beamformed_signal(j) + signals(i, j) * weights(i);
    end
end

% Display the beamformed signal
for j = 1:INPUT_LENGTH
    fprintf('Beamformed Signal[%d]: %f\n', j-1, beamformed_signal(j));
end
