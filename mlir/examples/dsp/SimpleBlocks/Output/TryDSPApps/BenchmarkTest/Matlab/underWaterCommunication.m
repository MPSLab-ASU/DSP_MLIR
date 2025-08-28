% Constants
PI = 3.14159265359;
FS = 1000;
INPUT_LENGTH = 40;
FILTER_ORDER = 5;

% Generate input time vector
input = (0:INPUT_LENGTH-1)' * 0.000125;

% Generate base sinusoid
getMultiplier = 2 * PI * 5;
getSinDuration = input * getMultiplier;
signal = sin(getSinDuration);

% Create delayed noise and add to signal
noise = [zeros(5,1); signal(1:end-5)];
noisy_sig = signal + noise;

% Filter configuration
wc = 2 * PI * 1000 / 500;
lpf = lowPassFIRFilter(wc);
hamming_window = hamming(FILTER_ORDER);

% Use first coefficient of Hamming window as in C
lpf_w = lpf * hamming_window(1);

% Apply scalar FIR filter
FIRfilterResponseArray = noisy_sig * lpf_w;

% Apply threshold
threshold = 0.05;
GetThresholdReal = thresholdUp(FIRfilterResponseArray, threshold, 0);

% Extract final result
final1 = GetThresholdReal(4); % MATLAB is 1-indexed
fprintf('%.6f\n', final1);
function out = lowPassFIRFilter(wc)
    PI = 3.14159265359;
    out = wc / PI;
end

function out = thresholdUp(input, threshold, defaultValue)
    out = double(input >= threshold);
    out(out < 1) = defaultValue;
end
