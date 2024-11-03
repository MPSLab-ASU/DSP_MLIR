% Define constants
INPUT_LENGTH = 1000;
SAMPLE_RATE = 8000;
step = 0.000125;
WINDOW_SIZE = 3;

% Generate input range
input = (0:step:(INPUT_LENGTH-1)*step)';

% Signal parameters
f_sig = 500;
f_noise = 3000;

% Generate clean signal
clean_sig = sin(2*pi*f_sig*input);

% Generate noise
noise = 0.5 * sin(2*pi*f_noise*input);

% Create noisy signal
noisy_sig = clean_sig + noise;

% Apply median filter
median_filtered = medfilt1(noisy_sig, WINDOW_SIZE);

% Apply moving average filter
avg_filtered = movmean(median_filtered, WINDOW_SIZE);

% Print the 4th element of the final result
disp(avg_filtered(4));
