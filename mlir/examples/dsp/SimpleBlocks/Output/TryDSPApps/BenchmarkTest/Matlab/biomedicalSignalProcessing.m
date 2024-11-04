% Constants
INPUT_LENGTH = 20000000;
MAX_PEAKS = 1000;
N = 101;

% Signal parameters
fc1 = 1000;
fc2 = 7500;
Fs = 8000;
distance = 950;
f_sig = 500;
f_noise = 3000;

% Generate input signal
t = (0:0.000125:(INPUT_LENGTH-1)*0.000125)';

% Generate clean signal
clean_sig = sin(2*pi*f_sig*t);

% Generate noise
noise = 0.5 * sin(2*pi*f_noise*t);

% Create noisy signal
noisy_sig = clean_sig + noise;

% Step 1: FIR Bandpass Filter
wc1 = 2 * pi * fc1 / Fs;
wc2 = 2 * pi * fc2 / Fs;

% Design lowpass filters
n = 0:N-1;
mid = (N-1)/2;
lpf1 = (wc1/pi) * sinc(wc1*(n-mid)/pi);
lpf2 = (wc2/pi) * sinc(wc2*(n-mid)/pi);

% Apply Hamming window
hamming_window = hamming(N)';
lpf1_w = lpf1 .* hamming_window;
lpf2_w = lpf2 .* hamming_window;

% Create bandpass filter
bpf_w = lpf2_w - lpf1_w;

% Apply bandpass filter
FIRfilterResponseForBpf = filter(bpf_w, 1, noisy_sig);

% Step 2: Artifact Removal (R-peak detection)
max_val = max(FIRfilterResponseForBpf);
height = 0.3 * max_val;

% Find peaks
[~, r_peaks] = findpeaks(FIRfilterResponseForBpf, 'MinPeakHeight', height, 'MinPeakDistance', distance);

% Calculate heart rate
diff_val = diff(r_peaks);
diff_mean = mean(diff_val);

avg_hr = (60 * Fs) / diff_mean;

fprintf('%f\n', avg_hr);
