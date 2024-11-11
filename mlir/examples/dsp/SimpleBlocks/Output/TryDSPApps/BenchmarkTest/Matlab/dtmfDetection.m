clc;
clear;

% Constants
fs = 8000;   % Sampling frequency
INPUT_LENGTH = 100000;  % Reduced from 100000000 to avoid excessive memory usage

% DTMF frequencies
dtmf_freqs = [697, 1209; 697, 1336; 697, 1477;
              770, 1209; 770, 1336; 770, 1477;
              852, 1209; 852, 1336; 852, 1477;
              941, 1336];

dtmf_digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"];  % Changed to string array

% Generate DTMF tone
function signal = generate_dtmf_tone(digit, N, fs, dtmf_digits, dtmf_freqs)
    idx = find(dtmf_digits == digit);
    if isempty(idx)
        error('InvalidDTMFDigit:NotFound', 'Invalid DTMF digit: %s', digit);
    end
    
    f1 = dtmf_freqs(idx, 1);
    f2 = dtmf_freqs(idx, 2);
    
    t = (0:N-1) / fs;
    signal = sin(2 * pi * f1 * t) + sin(2 * pi * f2 * t);
    signal = signal / max(abs(signal));  % Normalize to avoid clipping
end

% Goertzel algorithm (unchanged)
function magnitude = goertzel(data, N, frequency, fs)
    omega = 2.0 * pi * frequency / fs;
    cosine = cos(omega);
    coeff = 2.0 * cosine;
    q0 = 0;
    q1 = 0;
    q2 = 0;

    for i = 1:N
        q0 = coeff * q1 - q2 + data(i);
        q2 = q1;
        q1 = q0;
    end

    magnitude = sqrt(q1^2 + q2^2 - coeff * q1 * q2);
end

% Detect DTMF (unchanged)
function detected_digit = detect_dtmf(signal, N, fs, dtmf_digits, dtmf_freqs)
    frequencies = unique(dtmf_freqs(:));
    magnitudes = zeros(1, length(frequencies));

    % Calculate magnitudes for all DTMF frequencies
    for i = 1:length(frequencies)
        magnitudes(i) = goertzel(signal, N, frequencies(i), fs);
    end

    % Find the maximum magnitude in low and high frequency groups
    [~, max_low_index] = max(magnitudes(1:4));
    [~, max_high_index] = max(magnitudes(5:end));

    % Calculate the average magnitude
    avg_magnitude = mean(magnitudes);

    % Set threshold
    threshold = avg_magnitude * 2;

    % Check if the detected magnitudes are significantly above the threshold
    if magnitudes(max_low_index) > threshold && magnitudes(max_high_index + 4) > threshold
        detected_freqs = [frequencies(max_low_index), frequencies(max_high_index + 4)];
        [~, digit_index] = ismember(detected_freqs, dtmf_freqs, 'rows');
        if digit_index > 0
            detected_digit = dtmf_digits(digit_index);
        else
            detected_digit = "";
        end
    else
        detected_digit = "";
    end
end

% Main program
test_digit = "10";  % Test DTMF digit (changed to a non-DTMF digit for demonstration)
N = INPUT_LENGTH;
duration = N / fs;

try
    % Generate the DTMF tone
    input_signal = generate_dtmf_tone(test_digit, N, fs, dtmf_digits, dtmf_freqs);

    % Introduce delay
    delay_samples = round(fs / 100);
    padded_signal = [zeros(1, delay_samples), input_signal];
    input_signal = padded_signal(1:N);

    % Detect the DTMF tone
    detected_digit = detect_dtmf(input_signal, N, fs, dtmf_digits, dtmf_freqs);

    % Output results
    if ~isempty(detected_digit)
        fprintf('Generated: %s, Detected: %s\n', test_digit, detected_digit);
    else
        fprintf('Generated: %s, No DTMF digit detected\n', test_digit);
    end
catch ME
    if strcmp(ME.identifier, 'InvalidDTMFDigit:NotFound')
        fprintf('Error: %s is not a valid DTMF digit\n', test_digit);
    else
        rethrow(ME);
    end
end