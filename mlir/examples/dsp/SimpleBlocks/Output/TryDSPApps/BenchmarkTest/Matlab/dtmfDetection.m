INPUT_LENGTH = 1000;
digit = 8;
fs = 8192;
duration = INPUT_LENGTH / fs;
N = round(fs * duration);

dtmf_tone = generateDtmf(digit, fs, duration);

fft_real = dftReal(dtmf_tone);
fft_imag = dftImag(dtmf_tone);

magnitudes = sqrt(fft_real.^2 + fft_imag.^2);
frequencies = zeros(1, N);
for i = 1:N
    if i <= N/2
        frequencies(i) = (i-1) * fs / N;
    else
        frequencies(i) = (i-1 - N) * fs / N;
    end
end

peaks = findDominantPeaks(frequencies, magnitudes);

fprintf('%.6f %.6f\t', peaks(1), peaks(2));

freqPairs = [
    941, 1336; 697, 1209; 697, 1336; 697, 1477;
    770, 1209; 770, 1336; 770, 1477; 852, 1209;
    852, 1336; 852, 1477
];

recovered_digit = recoverDTMFDigit(peaks, freqPairs);
fprintf('%.6f\n', recovered_digit);

function tone = generateDtmf(digit, fs, duration)
    freqPairs = [
        941, 1336; 697, 1209; 697, 1336; 697, 1477;
        770, 1209; 770, 1336; 770, 1477; 852, 1209;
        852, 1336; 852, 1477
    ];
    f1 = freqPairs(digit + 1, 1);
    f2 = freqPairs(digit + 1, 2);
    N = fs * duration;
    t = (0:N-1) / fs;
    tone = 10 * sin(2 * pi * f1 * t) + sin(2 * pi * f2 * t);
end

function real = dftReal(input)
    N = length(input);
    real = zeros(1, N);
    for k = 0:N-1
        for n = 0:N-1
            angle = 2 * pi * k * n / N;
            real(k+1) = real(k+1) + input(n+1) * cos(angle);
        end
    end
end

function imag = dftImag(input)
    N = length(input);
    imag = zeros(1, N);
    for k = 0:N-1
        for n = 0:N-1
            angle = 2 * pi * k * n / N;
            imag(k+1) = imag(k+1) - input(n+1) * sin(angle);
        end
    end
end

function peaks = findDominantPeaks(frequencies, magnitudes)
    max1 = 0; max2 = 0;
    freq1 = 0; freq2 = 0;
    for i = 1:length(frequencies)
        currentFreq = frequencies(i);
        currentMag = magnitudes(i);
        if currentFreq >= 0
            if currentMag > max1
                max2 = max1;
                freq2 = freq1;
                max1 = currentMag;
                freq1 = currentFreq;
            elseif currentMag > max2
                max2 = currentMag;
                freq2 = currentFreq;
            end
        end
    end
    if freq1 < freq2
        peaks = [freq1, freq2];
    else
        peaks = [freq2, freq1];
    end
end

function digit = recoverDTMFDigit(peaks, freqPairs)
    digit = -1;
    for i = 1:size(freqPairs, 1)
        f1 = freqPairs(i, 1);
        f2 = freqPairs(i, 2);
        if (abs(peaks(1) - f1) < 10 && abs(peaks(2) - f2) < 10) || ...
           (abs(peaks(1) - f2) < 10 && abs(peaks(2) - f1) < 10)
            digit = i - 1;
            return;
        end
    end
end
