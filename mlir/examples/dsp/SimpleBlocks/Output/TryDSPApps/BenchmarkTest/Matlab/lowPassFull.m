% Define constants
PI = pi;
INPUT_LENGTH = 100000000;
fs = 8000;

% Generate input vector
input = (0:0.000125:(INPUT_LENGTH-1)*0.000125)';

% Signal processing steps
f_sig = 500;
getSinDuration = 2 * PI * f_sig * input;
clean_sig = sin(getSinDuration);

f_noise = 3000;
getNoiseSinDuration = 2 * PI * f_noise * input;
noise = sin(getNoiseSinDuration);

scaled_noise = 0.5 * noise;
noisy_sig = clean_sig + scaled_noise;

% Filter design
fc = 1000;
wc = 2 * PI * fc / fs;
N = 101;

% Low-pass FIR filter
n = -(N-1)/2:(N-1)/2;
lpf = (wc / PI) * sinc(wc * n / PI);

% Hamming window
hamming = 0.54 - 0.46 * cos(2 * PI * (0:N-1) / (N-1));

% Apply window to filter
lpf_w = lpf .* hamming;

% Apply FIR filter
FIRfilterResponse = filter(lpf_w, 1, noisy_sig);

% Display results
disp(FIRfilterResponse(2));