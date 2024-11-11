% Sampling frequency and duration
fs = 8192;  % Sampling frequency
duration = 0.5;  % Duration of the DTMF signal
N = fs * duration;  % Number of samples for the DTMF signal

% DTMF frequency pairs (same as in the C code)
freqPairs = [ 941, 1336;   % 0
              697, 1209;   % 1
              697, 1336;   % 2
              697, 1477;   % 3
              770, 1209;   % 4
              770, 1336;   % 5
              770, 1477;   % 6
              852, 1209;   % 7
              852, 1336;   % 8
              852, 1477];  % 9

% Function to generate the DTMF tone
function dtmf_tone = generateDtmf(digit, N, fs)
    freqPairs = [ 941, 1336; 697, 1209; 697, 1336; 697, 1477; 770, 1209; 
                  770, 1336; 770, 1477; 852, 1209; 852, 1336; 852, 1477];
    f1 = freqPairs(digit + 1, 1);
    f2 = freqPairs(digit + 1, 2);
    t = (0:N-1) / fs;
    dtmf_tone = 10 * (sin(2 * pi * f1 * t) + sin(2 * pi * f2 * t));
end

% Function to perform the Discrete Fourier Transform (DFT)
function [real_out, imag_out] = dft(signal, N)
    real_out = zeros(1, N);
    imag_out = zeros(1, N);
    for k = 1:N
        for n = 1:N
            angle = 2 * pi * (k-1) * (n-1) / N;
            real_out(k) = real_out(k) + signal(n) * cos(angle);
            imag_out(k) = imag_out(k) - signal(n) * sin(angle);
        end
    end
end

% Function to find the two highest peaks in the magnitude spectrum
function peaks = findDominantPeaks(frequencies, magnitudes)
    [~, idx] = sort(magnitudes, 'descend');  % Sort magnitudes in descending order
    idx1 = idx(1);
    idx2 = idx(2);
    peaks = sort([frequencies(idx1), frequencies(idx2)]);  % Sort frequencies in ascending order
end

% Function to recover the DTMF digit from the frequency peaks
function recovered_digit = recoverDtmfDigit(peaks, freqPairs)
    tolerance = 10;  % Tolerance for frequency matching
    recovered_digit = -1;  % Default value if no digit is detected
    for i = 1:10
        f1 = freqPairs(i, 1);
        f2 = freqPairs(i, 2);
        if (abs(peaks(1) - f1) < tolerance && abs(peaks(2) - f2) < tolerance) || ...
           (abs(peaks(1) - f2) < tolerance && abs(peaks(2) - f1) < tolerance)
            recovered_digit = i - 1;  % Match found, return digit
            return;
        end
    end
end

% Main program
digit = 9;  % DTMF digit to be generated
dtmf_tone = generateDtmf(digit, N, fs);  % Generate the DTMF tone

% Perform DFT (manually)
[real_out, imag_out] = dft(dtmf_tone, N);

% Calculate magnitudes and corresponding frequencies
magnitudes = sqrt(real_out.^2 + imag_out.^2);
frequencies = (0:(N-1)) * fs / N;

% Find dominant frequency peaks (in ascending order)
peaks = findDominantPeaks(frequencies(1:N/2), magnitudes(1:N/2));

% Recover the DTMF digit from the peaks
recovered_digit = recoverDtmfDigit(peaks, freqPairs);

% Display results
if recovered_digit >= 0
    fprintf('Recovered DTMF digit: %d\n', recovered_digit);
else
    fprintf('No DTMF digit detected.\n');
end
