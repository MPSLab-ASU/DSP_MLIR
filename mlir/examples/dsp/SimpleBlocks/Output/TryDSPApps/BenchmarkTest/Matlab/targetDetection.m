% Constants
INPUT_LENGTH = 100000000;
MAX_PEAKS = 100;

% Generate input range
input = (0:0.000125:(INPUT_LENGTH-1)*0.000125)';

% Generate signals
getMultiplier = 2 * pi * 10;
getSinDuration = input * getMultiplier;
sig1 = sin(getSinDuration);

getMultiplier2 = 2 * pi * 20;
getSinDuration2 = input * getMultiplier2;
sinsig2 = sin(getSinDuration2);
sig2 = 0.5 * sinsig2;

% Combine signals
signal = sig1 + sig2;

% Add delayed noise
noise = [zeros(5, 1); signal(1:end-5)];
noisy_sig = signal + noise;

% LMS Filter
mu = 0.01;
filterSize = 20;
y = lmsFilterResponse(noisy_sig, signal, mu, filterSize);

% Find peaks
[peaks, ~] = findpeaks(signal, 'MinPeakHeight', 1, 'MinPeakDistance', 50);

% Display results
fprintf('%d %d\n', peaks(2), peaks(3));


% LMS Filter Response Function
function output = lmsFilterResponse(noisy_sig, clean_sig, mu, filterSize)
    length = numel(noisy_sig);
    w = zeros(filterSize, 1);
    output = zeros(length, 1);
    
    for n = 1:length
        x = noisy_sig(max(1, n-filterSize+1):n);
        x = [zeros(filterSize - numel(x), 1); x];
        y = w' * x;
        e = clean_sig(n) - y;
        w = w + mu * e * x;
        output(n) = e;
    end
end