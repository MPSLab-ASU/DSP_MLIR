% Constants
INPUT_LENGTH = 10000000;
fs = 1000;

% Generate input signal
input = 0:(INPUT_LENGTH-1);

% Generate first sinusoidal signal
getMultiplier = 2 * pi * 50;
getSinDuration = input * getMultiplier;
sig1 = sin(getSinDuration);

% Generate second sinusoidal signal
getMultiplier2 = 2 * pi * 120;
getSinDuration2 = input * getMultiplier2;
sig2 = 0.5 * sin(getSinDuration2);

% Combine signals
signal = sig1 + sig2;

% Add delayed noise
noise = [zeros(1, 5), signal(1:end-5)];
noisy_sig = signal + noise;

% Perform DFT
dft_output = fft(noisy_sig);

% Calculate squared magnitude
sq_abs = abs(dft_output).^2;

% Calculate mean
res = mean(sq_abs);

% Apply threshold
threshold_value = 0.2;
GetThresholdReal = sq_abs .* (sq_abs >= threshold_value);

% Display results
disp(GetThresholdReal);
