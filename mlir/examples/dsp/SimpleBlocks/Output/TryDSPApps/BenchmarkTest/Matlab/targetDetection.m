% Constants
PI = pi;
INPUT_LENGTH = 500;
FILTER_SIZE = 20;
MAX_PEAKS = 50;

% Generate input time vector
input = 0:0.000125:(INPUT_LENGTH - 1) * 0.000125;

% Signal 1: 10 Hz sine
getMultiplier = 2 * PI * 10;
getSinDuration = getMultiplier * input;
sig1 = sin(getSinDuration);

% Signal 2: 20 Hz sine with 0.5 amplitude
getMultiplier2 = 2 * PI * 20;
getSinDuration2 = getMultiplier2 * input;
sinsig2 = sin(getSinDuration2);
sig2 = 0.5 * sinsig2;

% Combine signals
signal = sig1 + sig2;

% Create delayed noise (5-sample delay)
delaySamples = 5;
noise = [zeros(1, delaySamples), signal(1:end - delaySamples)];

% Add noise to signal
noisy_sig = signal + noise;

% LMS filtering using function
mu = 0.01;
y = lmsFilterResponse(noisy_sig, signal, mu, FILTER_SIZE);

% Peak detection
[~, peakLocs] = findpeaks(y, 'MinPeakHeight', 1.0, 'MinPeakDistance', 50);
numPeaks = min(length(peakLocs), MAX_PEAKS - 1);
peaks = -1 * ones(1, MAX_PEAKS);
peaks(1:numPeaks) = peakLocs(1:numPeaks);
peaks(end) = numPeaks;

% Extract two peak indices
final1 = peaks(2);
final2 = peaks(1);

fprintf('%f\t%f\n', final1, final2);
function y = lmsFilterResponse(noisy_sig, clean_sig, mu, filterSize)
    len = length(noisy_sig);
    y = zeros(1, len);
    w = zeros(1, filterSize);  % Initialize weights to zero

    for n = 1:len
        for i = 1:filterSize
            if (n - i + 1) > 0
                y(n) = y(n) + w(i) * noisy_sig(n - i + 1);
            end
        end
        e = clean_sig(n) - y(n);
        for i = 1:filterSize
            if (n - i + 1) > 0
                w(i) = w(i) + mu * e * noisy_sig(n - i + 1);
            end
        end
    end
end
