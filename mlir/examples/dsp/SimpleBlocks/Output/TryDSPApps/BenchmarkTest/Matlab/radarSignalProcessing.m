INPUT_LENGTH = 10;
fs = 8000;
fc1 = 1000;
fc2 = 7500;
N = 101;

input = getRangeOfVector(0, INPUT_LENGTH, 0.000125);
weights = getRangeOfVector(-90, 180, 1);
signal = beamForm(4, 5, input, weights);

b1 = abs(signal);
power = b1.^2;

wc1 = 2 * pi * fc1 / fs;
wc2 = 2 * pi * fc2 / fs;

filter1 = lowPassFIRFilter(wc1, N);
filter2 = highPassFIRFilter(wc2, N);

filter_hamming_1 = filter1 .* hammingWindow(N);
filter_hamming_2 = filter2 .* hammingWindow(N);

bpf = filter_hamming_2 - filter_hamming_1;
firResponse = firFilterResponse(power, bpf);

fprintf('%.6f\n', firResponse(11));

function vector = getRangeOfVector(first, N, step)
    vector = first + (0:N-1) * step;
end

function signal = beamForm(antennas, freq, time, weights)
    timeDim = length(time);
    signalMat = zeros(antennas, timeDim);
    for i = 1:antennas
        phase_shift = (i-1) * pi / 4;
        signalMat(i, :) = sin(2 * pi * freq * time + phase_shift);
    end
    signal = sum(signalMat(1:antennas, :) .* weights(1:antennas)', 1);
end

function filter = lowPassFIRFilter(wc, N)
    mid = floor(N/2);
    filter = zeros(1, N);
    for i = 1:N
        n = i - mid - 1;
        if n == 0
            filter(i) = wc / pi;
        else
            filter(i) = sin(wc * n) / (pi * n);
        end
    end
end

function filter = highPassFIRFilter(wc, N)
    mid = floor(N/2);
    filter = zeros(1, N);
    for i = 1:N
        n = i - mid - 1;
        if n == 0
            filter(i) = 1 - wc / pi;
        else
            filter(i) = -sin(wc * n) / (pi * n);
        end
    end
end

function window = hammingWindow(N)
    n = 0:N-1;
    window = 0.54 - 0.46 * cos(2 * pi * n / (N - 1));
end

function output = firFilterResponse(input, filter)
    outputLen = length(input) + length(filter) - 1;
    output = zeros(1, outputLen);
    for i = 1:outputLen
        for k = 1:length(filter)
            if i - k + 1 > 0 && i - k + 1 <= length(input)
                output(i) = output(i) + filter(k) * input(i - k + 1);
            end
        end
    end
end
