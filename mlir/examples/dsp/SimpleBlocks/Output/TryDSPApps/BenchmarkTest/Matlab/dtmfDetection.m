% DTMF Detection in MATLAB using DFT

% Constants
SAMPLING_FREQUENCY = 8192;  % Sampling frequency
DURATION = 0.5;             % Duration of the DTMF signal
N_SAMPLES = SAMPLING_FREQUENCY * DURATION; % Number of samples for the DTMF signal

% DTMF frequencies
freqPairs = [
    941, 1336;   % 0
    697, 1209;   % 1
    697, 1336;   % 2
    697, 1477;   % 3
    770, 1209;   % 4
    770, 1336;   % 5
    770, 1477;   % 6
    852, 1209;   % 7
    852, 1336;   % 8
    852, 1477    % 9
];

% Main script
digit = 0; % DTMF digit to be generated
fs = SAMPLING_FREQUENCY;
duration = DURATION;

% Generate the DTMF tone
dtmf_tone = generateDtmf(digit, fs, duration, freqPairs);

% Perform DFT
[real_out, imag_out] = dft(dtmf_tone);

% Calculate magnitudes and frequencies
N = length(dtmf_tone);
magnitudes = sqrt(real_out.^2 + imag_out.^2);
frequencies = (0:N-1)' * fs / N;
frequencies(frequencies > fs/2) = frequencies(frequencies > fs/2) - fs;

% Find dominant frequency peaks
peaks = findDominantPeaks(frequencies, magnitudes);

% Recover the DTMF digit
recovered_digit = recoverDtmfDigit(peaks, freqPairs);

% Display results
if recovered_digit >= 0
    fprintf('Recovered DTMF digit: %d\n', recovered_digit);
else
    fprintf('No DTMF digit detected.\n');
end

% Function definitions
function dtmf_tone = generateDtmf(digit, fs, duration, freqPairs)
    f1 = freqPairs(digit + 1, 1);
    f2 = freqPairs(digit + 1, 2);
    t = (0:1/fs:duration-1/fs)';
    dtmf_tone = 10 * (sin(2 * pi * f1 * t) + sin(2 * pi * f2 * t));
end

function [real_out, imag_out] = dft(signal)
    N = length(signal);
    real_out = zeros(N, 1);
    imag_out = zeros(N, 1);
    for k = 0:N-1
        for n = 0:N-1
            angle = 2 * pi * k * n / N;
            real_out(k+1) = real_out(k+1) + signal(n+1) * cos(angle);
            imag_out(k+1) = imag_out(k+1) - signal(n+1) * sin(angle);
        end
    end
end

function peaks = findDominantPeaks(frequencies, magnitudes)
    max1 = 0; max2 = 0;
    freq1 = 0; freq2 = 0;

    for i = 1:length(frequencies)
        currentFreq = frequencies(i);
        currentMag = magnitudes(i);

        % Check if frequency is positive
        if currentFreq >= 0
            % Compare current magnitude with max1
            if currentMag > max1
                % Update max2 and freq2 with previous max1 and freq1
                max2 = max1;
                freq2 = freq1;
                % Update max1 and freq1 with current values
                max1 = currentMag;
                freq1 = currentFreq;
            elseif currentMag > max2
                % Update max2 and freq2 with current values
                max2 = currentMag;
                freq2 = currentFreq;
            end
        end
    end

    % Compare freq1 and freq2 to determine the order
    if freq1 < freq2
        peaks = [freq1, freq2];
    else
        peaks = [freq2, freq1];
    end
end

function digit = recoverDtmfDigit(peaks, freqPairs)
    for i = 1:size(freqPairs, 1)
        f1 = freqPairs(i, 1);
        f2 = freqPairs(i, 2);

        if (abs(peaks(1) - f1) < 10 && abs(peaks(2) - f2) < 10) || ...
           (abs(peaks(1) - f2) < 10 && abs(peaks(2) - f1) < 10)
            digit = i - 1; % Digit found (subtract 1 because MATLAB is 1-indexed)
            return;
        end
    end
    digit = -1; % No match found
end