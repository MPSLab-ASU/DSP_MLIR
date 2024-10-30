% Constants
INPUT_LENGTH = 100000000;
PI = pi; % MATLAB has pi built-in
fs = 8000;
step = 1 / fs;

% Generate input range
input = (0:step:(INPUT_LENGTH-1)*step)';

% Generate clean signal
f_sig = 500;
clean_sig = sin(2 * PI * f_sig * input);

% Generate noise signal with a delay of 2 samples
noise = [zeros(2, 1); clean_sig(1:end-2)];

% Create noisy signal by adding noise to clean signal
noisy_sig = clean_sig + noise;

% LMS filter parameters
mu = 0.01;
filterSize = 32;

% LMS filter implementation
w = zeros(filterSize, 1);
y = zeros(INPUT_LENGTH, 1);

for n = filterSize:INPUT_LENGTH
    x = noisy_sig(n:-1:n-filterSize+1);
    y(n) = w' * x;
    e = clean_sig(n) - y(n);
    w = w + mu * e * x;
end

% Print result
fprintf('%f\n', y);
