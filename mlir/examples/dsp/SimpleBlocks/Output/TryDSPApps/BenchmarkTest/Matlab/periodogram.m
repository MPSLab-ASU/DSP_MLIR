% Define INPUT_LENGTH globally
INPUT_LENGTH = 10;

% Generate input range
input = 0:1:(INPUT_LENGTH-1);

% Reverse input
reverse_input = flip(input);

% FIR Filter Response (Convolution)
conv1d = conv(input, reverse_input, 'same');

% Compute DFT using FFT
fft_result = fft(conv1d);

% Compute square magnitude
sq = abs(fft_result).^2;

% Display results
fprintf('%f\n', sq);
