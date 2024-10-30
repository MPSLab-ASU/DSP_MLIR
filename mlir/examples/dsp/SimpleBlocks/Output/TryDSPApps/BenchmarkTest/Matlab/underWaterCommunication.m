% Constants
INPUT_LENGTH = 100000000;
FILTER_ORDER = 5;

% Sampling frequency
fs = 1000;

% Generate input vector
input = getRangeOfVector(0, INPUT_LENGTH, 1);

% Gain calculation
getMultiplier = 2 * pi * 5;
getSinDuration = gain(input, getMultiplier);

% Sine wave generation
signal = sine(getSinDuration);

% Adding delay (noise)
noise = delay(signal, 5);

% Adding signal and noise
noisy_sig = add(signal, noise);

% Low-pass filter parameters
fc = 1000;
wc = 2 * pi * fc / 500;  % wc should vary from 0 to pi

% Low-pass FIR filter design
lpf = lowPassFIRFilter(wc, FILTER_ORDER);
hamming_window = hamming(FILTER_ORDER);

% Apply Hamming window to the filter
lpf_w = lpf .* hamming_window;

% FIR filter response
FIRfilterResponse = FIRFilterResponse(noisy_sig, lpf_w);

% Thresholding operation
threshold = 0.5;
GetThresholdReal = thresholdUp(FIRfilterResponse, threshold, 0);

% Display the result
disp(GetThresholdReal(3));

% Function implementations

function vector = getRangeOfVector(start, length, increment)
    vector = (start : increment : start + (length-1)*increment)';
end

function output = gain(input, multiplier)
    output = input * multiplier;
end

function output = sine(input)
    output = sin(input);
end

function output = delay(input, delaySamples)
    output = [zeros(delaySamples, 1); input(1:end-delaySamples)];
end

function output = add(input1, input2)
    output = input1 + input2;
end

function filter = lowPassFIRFilter(wc, length)
    n = (-(length-1)/2:(length-1)/2)';
    filter = wc/pi * sinc(wc/pi * n);
end

function output = FIRFilterResponse(input, filter)
    output = conv(input, filter, 'same');
end

function output = thresholdUp(input, threshold, defaultValue)
    output = max(input, threshold);
    output(output == threshold) = defaultValue;
end