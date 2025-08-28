INPUT_LENGTH = 100;
FILTER_LENGTH = 200;
N = 101;
fs = 8000;

clean_signal = generateSignal(500, INPUT_LENGTH, fs);
noise_signal = generateSignal(3000, INPUT_LENGTH, fs);
noise_signal = noise_signal * 0.5;

noisy_signal = clean_signal + noise_signal;

fir_filter = generateLowpassFilter(1000, N, fs);
filtered_signal = applyFIRFilter(noisy_signal, fir_filter, FILTER_LENGTH, INPUT_LENGTH);

fprintf('%.6f\n', filtered_signal(7));

function signal = generateSignal(freq, length, fs)
    t = (0:length-1) / fs;
    signal = sin(2 * pi * freq * t);
end

function filter = generateLowpassFilter(cutoff_freq, N, fs)
    wc = 2 * pi * cutoff_freq / fs;
    mid = floor(N / 2);
    filter = zeros(1, N);
    for i = 1:N
        n = i - mid - 1;
        if n == 0
            filter(i) = wc / pi;
        else
            filter(i) = sin(wc * n) / (pi * n);
        end
        filter(i) = filter(i) * (0.54 - 0.46 * cos(2 * pi * (i-1) / (N - 1)));
    end
end

function output = applyFIRFilter(input, filter, out_length, input_length)
    N = length(filter);
    output = zeros(1, out_length);
    for i = 1:out_length
        sum = 0;
        for j = 1:N
            if i - j + 1 > 0 && i - j + 1 <= input_length
                sum = sum + input(i - j + 1) * filter(j);
            end
        end
        output(i) = sum;
    end
end